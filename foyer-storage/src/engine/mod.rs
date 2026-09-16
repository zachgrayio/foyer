// Copyright 2026 foyer Project Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    any::Any,
    fmt::Debug,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use foyer_common::{
    code::{StorageKey, StorageValue},
    error::Result,
    metrics::Metrics,
    properties::{Age, Properties},
    spawn::Spawner,
};
use foyer_memory::Piece;
use futures_core::future::BoxFuture;

use crate::{filter::StorageFilterResult, io::engine::IoEngine, keeper::PieceRef, Device};

use self::block::manager::{BlockSnapshot, ForceReclaimError};

/// A value-only snapshot of an entry's current disk address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryAddressSnapshot {
    /// Block containing the entry.
    pub block: u32,
    /// Byte offset of the entry within the block.
    pub offset: u32,
    /// Unaligned serialized entry length.
    pub len: u32,
    /// Monotonic entry version used to reject stale addresses.
    pub sequence: u64,
    /// Wall-clock time when this address was committed or recovered.
    pub inserted_at_unix_micros: u64,
    /// Wall-clock time when this address was last loaded.
    pub last_accessed_at_unix_micros: u64,
}

/// Opaque continuation for a mutation-detecting raw disk-index scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskIndexCursor {
    pub(crate) shard: usize,
    pub(crate) revision: u64,
    pub(crate) offset: usize,
}

/// A live hash/address pair from the raw disk index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskIndexEntry {
    /// Hash used by the disk index.
    pub hash: u64,
    /// Current disk address.
    pub address: EntryAddressSnapshot,
}

/// A bounded page from one raw disk-index shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiskIndexPage {
    /// Index shard represented by this page.
    pub shard: usize,
    /// Structural revision of the shard when this page was taken.
    pub revision: u64,
    /// Live addresses in this page, without decoded keys.
    pub entries: Vec<DiskIndexEntry>,
    /// Continuation for the next page, or `None` after the final shard.
    pub next_cursor: Option<DiskIndexCursor>,
    /// Whether prior observations for `shard` must be discarded and rescanned.
    pub retry_shard: bool,
}

impl From<block::indexer::EntryAddress> for EntryAddressSnapshot {
    fn from(address: block::indexer::EntryAddress) -> Self {
        Self {
            block: address.block,
            offset: address.offset,
            len: address.len,
            sequence: address.sequence,
            inserted_at_unix_micros: address.inserted_at_unix_micros,
            last_accessed_at_unix_micros: address.last_accessed_at_unix_micros.load(Ordering::Relaxed),
        }
    }
}

/// A decoded cache key and its current disk address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedEntry<K> {
    /// Hash used by the disk index.
    pub hash: u64,
    /// Decoded cache key.
    pub key: K,
    /// Current physical address of the entry.
    pub address: EntryAddressSnapshot,
}

/// A bounded page of decoded entries from the live disk index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedEntriesPage<K> {
    /// Number of live addresses in the disk index when the page was taken.
    pub total: usize,
    /// Raw index offset immediately after this page.
    pub next_offset: usize,
    /// Decoded entries in index iteration order.
    pub entries: Vec<InspectedEntry<K>>,
}

/// Why a live disk index entry was removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageRemovalReason {
    /// A newer entry replaced the same hash.
    Replaced,
    /// The key was explicitly deleted.
    Deleted,
    /// The containing block was reclaimed.
    Reclaimed,
    /// The indexed entry failed validation while being read.
    Invalid,
}

/// A structural change to the block storage engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageEvent {
    /// A new address became authoritative for a key hash.
    EntryCommitted {
        /// Indexed key hash.
        hash: u64,
        /// Committed disk address.
        address: EntryAddressSnapshot,
    },
    /// A live address stopped being authoritative.
    EntryRemoved {
        /// Indexed key hash.
        hash: u64,
        /// Removed disk address.
        address: EntryAddressSnapshot,
        /// Cause of removal.
        reason: StorageRemovalReason,
    },
    /// A live disk entry was loaded in a new reporting bucket.
    EntryAccessed {
        /// Indexed key hash.
        hash: u64,
        /// Current disk address and access timestamp.
        address: EntryAddressSnapshot,
    },
    /// A block changed lifecycle state.
    BlockStateChanged {
        /// Block identifier.
        block: u32,
        /// Process-local allocation generation.
        generation: u64,
        /// New lifecycle state.
        state: block::manager::BlockState,
    },
}

/// Non-blocking listener for structural storage events.
pub trait StorageEventListener: Send + Sync + 'static + Debug {
    /// Attempt to deliver an event, returning `false` if it was dropped.
    fn try_on_event(&self, event: StorageEvent) -> bool;
}

#[derive(Clone)]
pub(crate) struct StorageEventObserver {
    listener: Option<Arc<dyn StorageEventListener>>,
    dropped: Arc<AtomicU64>,
}

impl Debug for StorageEventObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageEventObserver")
            .field("enabled", &self.listener.is_some())
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl StorageEventObserver {
    pub(crate) fn new(listener: Option<Arc<dyn StorageEventListener>>) -> Self {
        Self {
            listener,
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    pub(crate) fn emit(&self, event: StorageEvent) {
        let delivered = self.listener.as_ref().is_none_or(|listener| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| listener.try_on_event(event))).unwrap_or(false)
        });
        if !delivered {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.listener.is_some()
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Source context for populated entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Populated {
    /// The age of the entry.
    pub age: Age,
}

/// Load result.
pub enum Load<K, V, P> {
    /// Load entry success.
    Entry {
        /// The key of the entry.
        key: K,
        /// The value of the entry.
        value: V,
        /// The populated context of the entry.
        populated: Populated,
    },
    /// Load entry success from disk cache write queue.
    Piece {
        /// The piece of the entry.
        piece: Piece<K, V, P>,
        /// The populated context of the entry.
        populated: Populated,
    },
    /// The entry may be in the disk cache, the read io is throttled.
    Throttled,
    /// Disk cache miss.
    Miss,
}

impl<K, V, P> Debug for Load<K, V, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Load::Entry { populated, .. } => f.debug_struct("Load::Entry").field("populated", populated).finish(),
            Load::Piece { piece, populated } => f
                .debug_struct("Load::Piece")
                .field("piece", piece)
                .field("populated", populated)
                .finish(),
            Load::Throttled => f.debug_struct("Load::Throttled").finish(),
            Load::Miss => f.debug_struct("Load::Miss").finish(),
        }
    }
}

impl<K, V, P> Load<K, V, P> {
    /// Return `Some` with the entry if load success, otherwise return `None`.
    pub fn entry(self) -> Option<(K, V, Populated)> {
        match self {
            Load::Entry { key, value, populated } => Some((key, value, populated)),
            _ => None,
        }
    }

    /// Return `Some` with the entry if load success, otherwise return `None`.
    ///
    /// Only key and value will be returned.
    pub fn kv(self) -> Option<(K, V)> {
        match self {
            Load::Entry { key, value, .. } => Some((key, value)),
            _ => None,
        }
    }

    /// Check if the load result is a cache miss.
    pub fn is_miss(&self) -> bool {
        matches!(self, Load::Miss)
    }

    /// Check if the load result is miss caused by io throttled.
    pub fn is_throttled(&self) -> bool {
        matches!(self, Load::Throttled)
    }
}

/// The recover mode of the disk cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum RecoverMode {
    /// Do not recover disk cache.
    ///
    /// For updatable cache, either [`RecoverMode::None`] or the tombstone log must be used to prevent from phantom
    /// entry when reopen.
    None,
    /// Recover disk cache and skip errors.
    #[default]
    Quiet,
    /// Recover disk cache and panic on errors.
    Strict,
}

/// Context for building the disk cache engine.
pub struct EngineBuildContext {
    /// IO engine for the disk cache engine.
    pub io_engine: Arc<dyn IoEngine>,
    /// Shared metrics for all components.
    pub metrics: Arc<Metrics>,
    /// The runtime for the disk cache engine.
    pub spawner: Spawner,
    /// The recover mode of the disk cache engine.
    pub recover_mode: RecoverMode,
}

/// Disk cache engine config trait.
#[expect(clippy::type_complexity)]
pub trait EngineConfig<K, V, P>: Send + Sync + 'static + Debug
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    /// Build the engine with the given configurations.
    fn build(self: Box<Self>, ctx: EngineBuildContext) -> BoxFuture<'static, Result<Arc<dyn Engine<K, V, P>>>>;

    /// Box the config.
    fn boxed(self) -> Box<Self>
    where
        Self: Sized,
    {
        Box::new(self)
    }
}

/// Disk cache engine trait.
pub trait Engine<K, V, P>: Send + Sync + 'static + Debug + Any
where
    K: StorageKey,
    V: StorageValue,
    P: Properties,
{
    /// Get the device used by this disk cache engine.
    fn device(&self) -> &Arc<dyn Device>;

    /// Return if the given key can be picked by the disk cache engine.
    fn filter(&self, hash: u64, estimated_size: usize) -> StorageFilterResult;

    /// Push a in-memory cache piece to the disk cache write queue.
    fn enqueue(&self, piece: PieceRef<K, V, P>, estimated_size: usize);

    /// Load a cache entry from the disk cache.
    ///
    /// `load` may return a false-positive result on entry key hash collision. It's the caller's responsibility to
    /// check if the returned key matches the given key.
    fn load(&self, hash: u64) -> BoxFuture<'static, Result<Load<K, V, P>>>;

    /// Delete the cache entry with the given key from the disk cache.
    fn delete(&self, hash: u64);

    /// Check if the disk cache contains a cached entry with the given key.
    ///
    /// `contains` may return a false-positive result if there is a hash collision with the given key.
    fn may_contains(&self, hash: u64) -> bool;

    /// Snapshot block lifecycle and occupancy information when supported.
    ///
    /// Lifecycle state and indexed occupancy are copied separately and can reflect adjacent moments during concurrent
    /// writes or reclamation.
    fn inspect_blocks(&self) -> Option<Vec<BlockSnapshot>> {
        None
    }

    /// Inspect the live entry currently indexed by `hash` when supported.
    fn inspect_entry(&self, _hash: u64) -> BoxFuture<'static, Result<Option<InspectedEntry<K>>>> {
        Box::pin(async { Ok(None) })
    }

    /// Decode the entry at an expected live address when supported.
    fn inspect_entry_at(
        &self,
        _hash: u64,
        _address: EntryAddressSnapshot,
    ) -> BoxFuture<'static, Result<Option<InspectedEntry<K>>>> {
        Box::pin(async { Ok(None) })
    }

    /// Snapshot the indexed address for `hash` without reading the entry key.
    fn inspect_address(&self, _hash: u64) -> Option<EntryAddressSnapshot> {
        None
    }

    /// Inspect one mutation-detecting page of raw live disk addresses when supported.
    fn inspect_disk_index_page(&self, _cursor: Option<DiskIndexCursor>, _limit: usize) -> Option<DiskIndexPage> {
        None
    }

    /// Inspect a bounded page of live decoded disk entries when supported.
    fn inspect_entries_page(
        &self,
        _offset: usize,
        _limit: usize,
    ) -> BoxFuture<'static, Result<Option<InspectedEntriesPage<K>>>> {
        Box::pin(async { Ok(None) })
    }

    /// Inspect all live entries currently indexed in `block` when supported.
    fn inspect_block(&self, _block: u32) -> BoxFuture<'static, Result<Option<Vec<InspectedEntry<K>>>>> {
        Box::pin(async { Ok(None) })
    }

    /// Reclaim an exact block if its process-local generation still matches.
    fn force_reclaim(
        &self,
        _block: u32,
        _expected_generation: u64,
    ) -> BoxFuture<'static, std::result::Result<(), ForceReclaimError>> {
        Box::pin(async { Err(ForceReclaimError::Unsupported) })
    }

    /// Return the number of structural events rejected by the listener.
    fn dropped_inspection_events(&self) -> u64 {
        0
    }

    /// Delete all cached entries of the disk cache.
    fn destroy(&self) -> BoxFuture<'static, Result<()>>;

    /// Wait for the ongoing flush and reclaim tasks to finish.
    fn wait(&self) -> BoxFuture<'static, ()>;

    /// Close the disk cache gracefully.
    ///
    /// `close` will wait for all ongoing flush and reclaim tasks to finish.
    fn close(&self) -> BoxFuture<'static, Result<()>>;
}

pub mod block;
pub mod noop;
