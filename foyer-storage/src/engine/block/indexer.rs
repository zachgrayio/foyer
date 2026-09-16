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
    collections::{hash_map::Entry, HashMap},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use itertools::Itertools;
use parking_lot::RwLock;

use crate::engine::{
    block::{manager::BlockId, serde::Sequence},
    DiskIndexCursor, DiskIndexEntry, DiskIndexPage,
};

#[derive(Debug, Clone)]
pub enum Index {
    Address(EntryAddress),
    Tombstone(Sequence),
}

impl Index {
    fn sequence(&self) -> Sequence {
        match self {
            Index::Address(addr) => addr.sequence,
            Index::Tombstone(seq) => *seq,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashedEntryAddress {
    pub hash: u64,
    pub address: EntryAddress,
}

#[derive(Debug, Clone)]
pub struct EntryAddress {
    pub block: BlockId,
    pub offset: u32,
    pub len: u32,

    pub sequence: Sequence,
    pub inserted_at_unix_micros: u64,
    pub last_accessed_at_unix_micros: Arc<AtomicU64>,
    pub last_access_report_bucket: Arc<AtomicU64>,
}

impl PartialEq for EntryAddress {
    fn eq(&self, other: &Self) -> bool {
        self.block == other.block
            && self.offset == other.offset
            && self.len == other.len
            && self.sequence == other.sequence
    }
}

impl Eq for EntryAddress {}

pub(crate) fn unix_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}

type IndexerShard = HashMap<u64, Index>;

/// [`Indexer`] records key hash to entry address on fs.
#[derive(Debug, Clone)]
pub struct Indexer {
    shards: Arc<Vec<RwLock<IndexerShard>>>,
    revisions: Arc<Vec<AtomicU64>>,
}

impl Indexer {
    pub fn new(shards: usize) -> Self {
        let revisions = (0..shards).map(|_| AtomicU64::new(0)).collect_vec();
        let shards = (0..shards).map(|_| RwLock::new(HashMap::new())).collect_vec();
        Self {
            shards: Arc::new(shards),
            revisions: Arc::new(revisions),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::insert_tombstone")
    )]
    pub fn insert_tombstone(&self, hash: u64, sequence: Sequence) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        let mut shard = self.shards[shard].write();
        let changed = shard.get(&hash).is_none_or(|index| sequence >= index.sequence());
        let old = self.insert_inner(&mut shard, hash, Index::Tombstone(sequence));
        if changed {
            self.revisions[self.shard(hash)].fetch_add(1, Ordering::Relaxed);
        }
        old
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::insert_batch")
    )]
    pub fn insert_batch(&self, batch: Vec<HashedEntryAddress>) -> Vec<HashedEntryAddress> {
        let shards: HashMap<usize, Vec<HashedEntryAddress>> =
            batch.into_iter().into_group_map_by(|haddr| self.shard(haddr.hash));

        let mut olds = vec![];
        for (s, batch) in shards {
            let mut shard = self.shards[s].write();
            let mut changed = false;
            for haddr in batch {
                changed |= shard
                    .get(&haddr.hash)
                    .is_none_or(|index| haddr.address.sequence >= index.sequence());
                if let Some(old) = self.insert_inner(&mut shard, haddr.hash, Index::Address(haddr.address)) {
                    olds.push(HashedEntryAddress {
                        hash: haddr.hash,
                        address: old,
                    });
                }
            }
            if changed {
                self.revisions[s].fetch_add(1, Ordering::Relaxed);
            }
        }
        olds
    }

    #[cfg_attr(feature = "tracing", fastrace::trace(name = "foyer::storage::block::indexer::get"))]
    pub fn get(&self, hash: u64) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        match self.shards[shard].read().get(&hash) {
            Some(index) => match index {
                Index::Address(addr) => {
                    addr.last_accessed_at_unix_micros
                        .store(unix_micros(), Ordering::Relaxed);
                    Some(addr.clone())
                }
                Index::Tombstone(_) => None,
            },
            None => None,
        }
    }

    /// Locate the live entry address currently indexed for `hash`.
    pub fn locate(&self, hash: u64) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        match self.shards[shard].read().get(&hash) {
            Some(Index::Address(address)) => Some(address.clone()),
            Some(Index::Tombstone(_)) | None => None,
        }
    }

    /// Snapshot a bounded page of live entry addresses without cloning skipped entries.
    pub fn snapshot_page(&self, offset: usize, limit: usize) -> (usize, Vec<HashedEntryAddress>) {
        let total = self
            .shards
            .iter()
            .map(|shard| {
                shard
                    .read()
                    .values()
                    .filter(|index| matches!(index, Index::Address(_)))
                    .count()
            })
            .sum();
        let entries = self
            .shards
            .iter()
            .flat_map(|shard| {
                shard
                    .read()
                    .iter()
                    .filter_map(|(&hash, index)| match index {
                        Index::Address(address) => Some(HashedEntryAddress {
                            hash,
                            address: address.clone(),
                        }),
                        Index::Tombstone(_) => None,
                    })
                    .collect_vec()
            })
            .skip(offset)
            .take(limit)
            .collect();
        (total, entries)
    }

    /// Snapshot a bounded page from one shard and detect structural mutations between pages.
    pub fn snapshot_cursor(&self, cursor: Option<DiskIndexCursor>, limit: usize) -> DiskIndexPage {
        assert!(limit > 0, "indexer cursor page limit must be greater than zero");

        let cursor = cursor.unwrap_or_else(|| DiskIndexCursor {
            shard: 0,
            revision: self.revisions[0].load(Ordering::Relaxed),
            offset: 0,
        });
        let shard = self.shards[cursor.shard].read();
        let revision = self.revisions[cursor.shard].load(Ordering::Relaxed);
        if cursor.revision != revision {
            return DiskIndexPage {
                shard: cursor.shard,
                revision,
                entries: vec![],
                next_cursor: Some(DiskIndexCursor {
                    shard: cursor.shard,
                    revision,
                    offset: 0,
                }),
                retry_shard: true,
            };
        }

        let mut entries = shard
            .iter()
            .filter_map(|(&hash, index)| match index {
                Index::Address(address) => Some(DiskIndexEntry {
                    hash,
                    address: address.clone().into(),
                }),
                Index::Tombstone(_) => None,
            })
            .skip(cursor.offset)
            .take(limit + 1)
            .collect_vec();
        let has_more = entries.len() > limit;
        entries.truncate(limit);
        let next = if has_more {
            Some(DiskIndexCursor {
                shard: cursor.shard,
                revision,
                offset: cursor.offset + entries.len(),
            })
        } else if cursor.shard + 1 < self.shards.len() {
            let next_shard = cursor.shard + 1;
            Some(DiskIndexCursor {
                shard: next_shard,
                revision: self.revisions[next_shard].load(Ordering::Relaxed),
                offset: 0,
            })
        } else {
            None
        };

        DiskIndexPage {
            shard: cursor.shard,
            revision,
            entries,
            next_cursor: next,
            retry_shard: false,
        }
    }

    /// Snapshot all live entry addresses currently indexed in `block`.
    pub fn snapshot_block(&self, block: BlockId) -> Vec<HashedEntryAddress> {
        self.shards
            .iter()
            .flat_map(|shard| {
                shard
                    .read()
                    .iter()
                    .filter_map(|(&hash, index)| match index {
                        Index::Address(address) if address.block == block => Some(HashedEntryAddress {
                            hash,
                            address: address.clone(),
                        }),
                        Index::Address(_) | Index::Tombstone(_) => None,
                    })
                    .collect_vec()
            })
            .collect()
    }

    /// Snapshot all live entry addresses currently held by the indexer.
    pub fn snapshot_all(&self) -> Vec<HashedEntryAddress> {
        self.shards
            .iter()
            .flat_map(|shard| {
                shard
                    .read()
                    .iter()
                    .filter_map(|(&hash, index)| match index {
                        Index::Address(address) => Some(HashedEntryAddress {
                            hash,
                            address: address.clone(),
                        }),
                        Index::Tombstone(_) => None,
                    })
                    .collect_vec()
            })
            .collect()
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::remove")
    )]
    pub fn remove(&self, hash: u64) -> Option<EntryAddress> {
        let shard = self.shard(hash);
        let mut guard = self.shards[shard].write();
        let removed = match guard.entry(hash) {
            Entry::Occupied(o) => match o.get() {
                Index::Address(_) => self.extract_address(o.remove()),
                Index::Tombstone(_) => None,
            },
            Entry::Vacant(_) => None,
        };
        if removed.is_some() {
            self.revisions[shard].fetch_add(1, Ordering::Relaxed);
        }
        removed
    }

    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "foyer::storage::block::indexer::remove_batch")
    )]
    pub fn remove_batch<I>(&self, batch: I) -> Vec<EntryAddress>
    where
        I: IntoIterator<Item = (u64, Sequence)>,
    {
        self.remove_batch_hashed(batch)
            .into_iter()
            .map(|entry| entry.address)
            .collect()
    }

    pub fn remove_batch_hashed<I>(&self, batch: I) -> Vec<HashedEntryAddress>
    where
        I: IntoIterator<Item = (u64, Sequence)>,
    {
        let shards = batch.into_iter().into_group_map_by(|(hash, _)| self.shard(*hash));

        let mut olds = vec![];
        for (s, hashes) in shards {
            let mut shard = self.shards[s].write();
            let mut changed = false;
            for (hash, sequence) in hashes {
                match shard.entry(hash) {
                    Entry::Occupied(o) => {
                        if sequence >= o.get().sequence() {
                            if let Some(addr) = self.extract_address(o.remove()) {
                                olds.push(HashedEntryAddress { hash, address: addr });
                                changed = true;
                            }
                        }
                    }
                    Entry::Vacant(_) => {}
                }
            }
            if changed {
                self.revisions[s].fetch_add(1, Ordering::Relaxed);
            }
        }
        olds
    }

    #[cfg_attr(feature = "tracing", fastrace::trace(name = "foyer::storage::block::indexer::clear"))]
    pub fn clear(&self) {
        for (index, shard) in self.shards.iter().enumerate() {
            let mut shard = shard.write();
            if !shard.is_empty() {
                shard.clear();
                self.revisions[index].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    fn shard(&self, hash: u64) -> usize {
        hash as usize % self.shards.len()
    }

    fn insert_inner(&self, shard: &mut IndexerShard, hash: u64, index: Index) -> Option<EntryAddress> {
        match shard.entry(hash) {
            Entry::Occupied(mut o) => {
                // `>` for updates.
                // '=' for reinsertions.
                if index.sequence() >= o.get().sequence() {
                    self.extract_address(o.insert(index))
                } else {
                    self.extract_address(index)
                }
            }
            Entry::Vacant(v) => {
                v.insert(index);
                None
            }
        }
    }

    fn extract_address(&self, index: Index) -> Option<EntryAddress> {
        match index {
            Index::Address(addr) => Some(addr),
            Index::Tombstone(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn address(block: BlockId, offset: u32, sequence: Sequence) -> EntryAddress {
        let now = unix_micros();
        EntryAddress {
            block,
            offset,
            len: 4096,
            sequence,
            inserted_at_unix_micros: now,
            last_accessed_at_unix_micros: Arc::new(AtomicU64::new(now)),
            last_access_report_bucket: Arc::new(AtomicU64::new(0)),
        }
    }

    #[test]
    fn snapshots_only_current_live_addresses() {
        let indexer = Indexer::new(4);
        indexer.insert_batch(vec![
            HashedEntryAddress {
                hash: 1,
                address: address(7, 4096, 1),
            },
            HashedEntryAddress {
                hash: 2,
                address: address(8, 4096, 2),
            },
        ]);

        indexer.insert_batch(vec![HashedEntryAddress {
            hash: 1,
            address: address(8, 8192, 3),
        }]);
        indexer.insert_tombstone(2, 4);

        assert!(indexer.snapshot_block(7).is_empty());
        let entries = indexer.snapshot_block(8);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].hash, 1);
        assert_eq!(entries[0].address, address(8, 8192, 3));
        assert_eq!(indexer.snapshot_all(), entries);
        assert_eq!(indexer.locate(1), Some(address(8, 8192, 3)));
    }

    #[test]
    fn cursor_retries_mutated_shard() {
        let indexer = Indexer::new(1);
        indexer.insert_batch(vec![
            HashedEntryAddress {
                hash: 1,
                address: address(1, 4096, 1),
            },
            HashedEntryAddress {
                hash: 2,
                address: address(1, 8192, 2),
            },
        ]);

        let first = indexer.snapshot_cursor(None, 1);
        assert!(!first.retry_shard);
        assert_eq!(first.entries.len(), 1);

        indexer.insert_batch(vec![HashedEntryAddress {
            hash: 3,
            address: address(1, 12_288, 3),
        }]);
        let retry = indexer.snapshot_cursor(first.next_cursor, 1);
        assert!(retry.retry_shard);
        assert!(retry.entries.is_empty());

        let mut cursor = retry.next_cursor;
        let mut hashes = vec![];
        while let Some(current) = cursor {
            let page = indexer.snapshot_cursor(Some(current), 1);
            assert!(!page.retry_shard);
            hashes.extend(page.entries.into_iter().map(|entry| entry.hash));
            cursor = page.next_cursor;
        }
        hashes.sort_unstable();
        assert_eq!(hashes, vec![1, 2, 3]);
    }
}
