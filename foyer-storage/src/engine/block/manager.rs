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
    collections::{HashSet, VecDeque},
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, RwLock, RwLockWriteGuard,
    },
};

use foyer_common::{
    error::{ErrorKind, Result},
    metrics::Metrics,
    spawn::Spawner,
};
use futures_core::future::BoxFuture;
use futures_util::{
    future::{ready, Shared},
    FutureExt,
};
use itertools::Itertools;
use mea::oneshot;
use rand::seq::IteratorRandom;

use crate::{
    engine::block::{
        eviction::{EvictionInfo, EvictionPicker},
        reclaimer::ReclaimerTrait,
    },
    engine::{StorageEvent, StorageEventObserver},
    io::{
        bytes::{IoB, IoBuf, IoBufMut},
        device::Partition,
        engine::IoEngine,
    },
    Device,
};

pub type BlockId = u32;

/// The current lifecycle state of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// The block has not been classified during recovery yet.
    Initializing,
    /// The block contains no entries and is available for writing.
    Clean,
    /// The block is currently receiving entries.
    Writing,
    /// The block contains entries and can be selected for reclamation.
    Evictable,
    /// The block is currently being reclaimed.
    Reclaiming,
}

/// A validation failure while requesting explicit block reclamation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForceReclaimError {
    /// The configured storage engine does not support block reclamation.
    Unsupported,
    /// The requested block identifier does not exist.
    OutOfRange {
        /// Requested block identifier.
        block: BlockId,
        /// Number of blocks in the engine.
        blocks: usize,
    },
    /// The block has been reused since the caller's snapshot.
    StaleGeneration {
        /// Generation supplied by the caller.
        expected: u64,
        /// Current block generation.
        actual: u64,
    },
    /// Only evictable blocks can be explicitly reclaimed.
    NotEvictable {
        /// Current lifecycle state of the block.
        state: BlockState,
    },
}

impl Display for ForceReclaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => write!(f, "block reclamation is not supported by this storage engine"),
            Self::OutOfRange { block, blocks } => {
                write!(f, "block {block} is out of range for an engine with {blocks} blocks")
            }
            Self::StaleGeneration { expected, actual } => {
                write!(f, "block generation changed from {expected} to {actual}")
            }
            Self::NotEvictable { state } => write!(f, "block is {state:?}, not evictable"),
        }
    }
}

impl std::error::Error for ForceReclaimError {}

/// A consistent, read-only snapshot of a block's physical state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSnapshot {
    /// Block identifier within this engine.
    pub id: BlockId,
    /// Total block capacity in bytes.
    pub size: usize,
    /// Current lifecycle state.
    pub state: BlockState,
    /// Process-local allocation generation.
    pub generation: u64,
    /// Number of entries currently pointing into this block.
    pub live_entries: usize,
    /// Page-aligned bytes occupied by currently indexed entries.
    pub live_bytes: usize,
    /// Estimated bytes invalidated by replacement or deletion.
    pub invalid_bytes: usize,
    /// Number of accesses recorded for this block.
    pub accesses: usize,
    /// Whether the block is marked for eviction probation.
    pub probation: bool,
}

/// Block statistics.
#[derive(Debug, Default)]
pub struct BlockStatistics {
    /// Estimated invalid bytes in the block.
    /// FIXME(MrCroxx): This value is way too coarse. Need fix.
    pub invalid: AtomicUsize,
    /// Access count of the block.
    pub access: AtomicUsize,
    /// Marked as `true` if the block is about to be evicted by some eviction picker.
    pub probation: AtomicBool,
}

impl BlockStatistics {
    pub(crate) fn reset(&self) {
        self.invalid.store(0, Ordering::Relaxed);
        self.access.store(0, Ordering::Relaxed);
        self.probation.store(false, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct BlockInner {
    id: BlockId,
    partition: Arc<dyn Partition>,
    io_engine: Arc<dyn IoEngine>,
    statistics: Arc<BlockStatistics>,
}

/// A block is a logical partition of a device. It is used to manage the device's storage space.
#[derive(Debug, Clone)]
pub struct Block {
    inner: Arc<BlockInner>,
}

impl Block {
    /// Get block id.
    pub fn id(&self) -> BlockId {
        self.inner.id
    }

    /// Get block Statistics.
    pub fn statistics(&self) -> &Arc<BlockStatistics> {
        &self.inner.statistics
    }

    /// Get block size.
    pub fn size(&self) -> usize {
        self.inner.partition.size()
    }

    pub(crate) async fn write(&self, buf: Box<dyn IoBuf>, offset: u64) -> (Box<dyn IoB>, Result<()>) {
        let (buf, res) = self
            .inner
            .io_engine
            .write(buf, self.inner.partition.as_ref(), offset)
            .await;
        (buf, res)
    }

    pub(crate) async fn read(&self, buf: Box<dyn IoBufMut>, offset: u64) -> (Box<dyn IoB>, Result<()>) {
        let (buf, res) = self
            .inner
            .io_engine
            .read(buf, self.inner.partition.as_ref(), offset)
            .await;
        (buf, res)
    }

    pub(crate) fn partition(&self) -> &Arc<dyn Partition> {
        &self.inner.partition
    }
}

#[cfg(test)]
impl Block {
    pub(crate) fn new_for_test(id: BlockId, partition: Arc<dyn Partition>, io_engine: Arc<dyn IoEngine>) -> Self {
        let inner = BlockInner {
            id,
            partition,
            io_engine,
            statistics: Arc::<BlockStatistics>::default(),
        };
        let inner = Arc::new(inner);
        Self { inner }
    }
}

pub type GetCleanBlockHandle = Shared<BoxFuture<'static, Block>>;

#[derive(Debug)]
struct State {
    clean_blocks: VecDeque<BlockId>,
    evictable_blocks: HashSet<BlockId>,
    writing_blocks: HashSet<BlockId>,
    reclaiming_blocks: HashSet<BlockId>,

    clean_block_waiters: Vec<oneshot::Sender<Block>>,

    eviction_pickers: Vec<Box<dyn EvictionPicker>>,

    reclaim_waiters: Vec<oneshot::Sender<()>>,
}

#[derive(Debug)]
struct Inner {
    blocks: Vec<Block>,
    generations: Vec<AtomicU64>,
    state: RwLock<State>,
    reclaimer: Arc<dyn ReclaimerTrait>,
    reclaim_concurrency: usize,
    clean_block_threshold: usize,
    metrics: Arc<Metrics>,
    spawner: Spawner,
    observer: StorageEventObserver,
}

#[derive(Debug, Clone)]
pub struct BlockManager {
    inner: Arc<Inner>,
}

impl BlockManager {
    fn block_state(state: &State, id: BlockId) -> BlockState {
        if state.clean_blocks.contains(&id) {
            BlockState::Clean
        } else if state.writing_blocks.contains(&id) {
            BlockState::Writing
        } else if state.evictable_blocks.contains(&id) {
            BlockState::Evictable
        } else if state.reclaiming_blocks.contains(&id) {
            BlockState::Reclaiming
        } else {
            BlockState::Initializing
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub fn open(
        device: Arc<dyn Device>,
        io_engine: Arc<dyn IoEngine>,
        block_size: usize,
        mut eviction_pickers: Vec<Box<dyn EvictionPicker>>,
        reclaimer: Arc<dyn ReclaimerTrait>,
        reclaim_concurrency: usize,
        clean_block_threshold: usize,
        metrics: Arc<Metrics>,
        spawner: Spawner,
        observer: StorageEventObserver,
    ) -> Result<Self> {
        let mut blocks = vec![];

        while device.free() >= block_size {
            let partition = match device.create_partition(block_size) {
                Ok(partition) => partition,
                Err(e) if e.kind() == ErrorKind::NoSpace => break,
                Err(e) => return Err(e),
            };
            let id = blocks.len() as BlockId;
            let block = Block {
                inner: Arc::new(BlockInner {
                    id,
                    partition,
                    io_engine: io_engine.clone(),
                    statistics: Arc::<BlockStatistics>::default(),
                }),
            };
            blocks.push(block);
        }

        let rs = blocks.iter().map(|r| r.id()).collect_vec();
        for pickers in eviction_pickers.iter_mut() {
            pickers.init(&rs, block_size);
        }

        metrics.storage_block_engine_block_size_bytes.absolute(block_size as _);

        let state = State {
            clean_blocks: VecDeque::new(),
            evictable_blocks: HashSet::new(),
            writing_blocks: HashSet::new(),
            reclaiming_blocks: HashSet::new(),
            clean_block_waiters: Vec::new(),
            eviction_pickers,
            reclaim_waiters: Vec::new(),
        };
        let inner = Inner {
            generations: (0..blocks.len()).map(|_| AtomicU64::new(0)).collect(),
            blocks,
            state: RwLock::new(state),
            reclaimer,
            reclaim_concurrency,
            clean_block_threshold,
            metrics,
            spawner,
            observer,
        };
        let inner = Arc::new(inner);
        let this = Self { inner };
        Ok(this)
    }

    pub fn init(&self, clean_blocks: &[BlockId]) {
        let mut state = self.inner.state.write().unwrap();
        let mut evictable_blocks: HashSet<BlockId> = self.inner.blocks.iter().map(|r| r.id()).collect();
        state.clean_blocks = clean_blocks
            .iter()
            .inspect(|id| {
                evictable_blocks.remove(id);
            })
            .copied()
            .collect();

        // Temporarily take pickers to make borrow checker happy.
        let mut pickers = std::mem::take(&mut state.eviction_pickers);

        // Notify pickers.
        for block in evictable_blocks {
            state.evictable_blocks.insert(block);
            for picker in pickers.iter_mut() {
                picker.on_block_evictable(
                    EvictionInfo {
                        blocks: &self.inner.blocks,
                        evictable: &state.evictable_blocks,
                        clean: state.clean_blocks.len(),
                    },
                    block,
                );
            }
        }

        // Restore taken pickers after operations.

        std::mem::swap(&mut state.eviction_pickers, &mut pickers);
        assert!(pickers.is_empty());

        let metrics = &self.inner.metrics;
        metrics
            .storage_block_engine_block_clean
            .absolute(state.clean_blocks.len() as _);
        metrics
            .storage_block_engine_block_evictable
            .absolute(state.evictable_blocks.len() as _);
        metrics
            .storage_block_engine_block_writing
            .absolute(state.writing_blocks.len() as _);
        metrics
            .storage_block_engine_block_reclaiming
            .absolute(state.reclaiming_blocks.len() as _);
    }

    pub fn blocks(&self) -> usize {
        self.inner.blocks.len()
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.inner.blocks[id as usize]
    }

    /// Snapshot the lifecycle state and statistics for every block.
    pub fn snapshots(&self) -> Vec<BlockSnapshot> {
        let state = self.inner.state.read().unwrap();
        self.inner
            .blocks
            .iter()
            .map(|block| {
                let id = block.id();
                let state = Self::block_state(&state, id);
                let statistics = block.statistics();
                BlockSnapshot {
                    id,
                    size: block.size(),
                    state,
                    generation: self.inner.generations[id as usize].load(Ordering::Relaxed),
                    live_entries: 0,
                    live_bytes: 0,
                    invalid_bytes: statistics.invalid.load(Ordering::Relaxed),
                    accesses: statistics.access.load(Ordering::Relaxed),
                    probation: statistics.probation.load(Ordering::Relaxed),
                }
            })
            .collect()
    }

    /// Reclaim an exact evictable block if its generation still matches the caller's snapshot.
    pub fn force_reclaim(&self, id: BlockId, expected_generation: u64) -> std::result::Result<(), ForceReclaimError> {
        let block = {
            let mut state = self.inner.state.write().unwrap();
            if id as usize >= self.inner.blocks.len() {
                return Err(ForceReclaimError::OutOfRange {
                    block: id,
                    blocks: self.inner.blocks.len(),
                });
            }

            let generation = self.inner.generations[id as usize].load(Ordering::Relaxed);
            if generation != expected_generation {
                return Err(ForceReclaimError::StaleGeneration {
                    expected: expected_generation,
                    actual: generation,
                });
            }
            if !state.evictable_blocks.remove(&id) {
                return Err(ForceReclaimError::NotEvictable {
                    state: Self::block_state(&state, id),
                });
            }

            self.inner.metrics.storage_block_engine_block_evictable.decrease(1);
            let mut pickers = std::mem::take(&mut state.eviction_pickers);
            for picker in pickers.iter_mut() {
                picker.on_block_evict(
                    EvictionInfo {
                        blocks: &self.inner.blocks,
                        evictable: &state.evictable_blocks,
                        clean: state.clean_blocks.len(),
                    },
                    id,
                );
            }
            std::mem::swap(&mut state.eviction_pickers, &mut pickers);
            assert!(pickers.is_empty());

            state.reclaiming_blocks.insert(id);
            self.inner.metrics.storage_block_engine_block_reclaiming.increase(1);
            ReclaimingBlock {
                block_manager: self.clone(),
                block: self.inner.blocks[id as usize].clone(),
                reinsert: false,
            }
        };
        self.start_reclaim(block);
        Ok(())
    }

    pub fn get_clean_block(&self) -> GetCleanBlockHandle {
        let this = self.clone();
        async move {
            // Wrap state lock guard to make borrow checker happy.
            let rx = {
                let mut state = this.inner.state.write().unwrap();
                if let Some(id) = state.clean_blocks.pop_front() {
                    let block = this.inner.blocks[id as usize].clone();
                    let generation = this.inner.generations[id as usize].fetch_add(1, Ordering::Relaxed) + 1;
                    state.writing_blocks.insert(id);
                    this.inner.metrics.storage_block_engine_block_clean.decrease(1);
                    this.inner.metrics.storage_block_engine_block_writing.increase(1);
                    let reclaiming = this.reclaim_if_needed(&mut state);
                    drop(state);
                    this.inner.observer.emit(StorageEvent::BlockStateChanged {
                        block: id,
                        generation,
                        state: BlockState::Writing,
                    });
                    if let Some(reclaiming) = reclaiming {
                        this.start_reclaim(reclaiming);
                    }
                    return block;
                } else {
                    let (tx, rx) = oneshot::channel();
                    state.clean_block_waiters.push(tx);
                    drop(state);
                    rx
                }
            };
            rx.await.unwrap()
        }
        .boxed()
        .shared()
    }

    pub fn on_writing_finish(&self, block: Block) {
        let mut state = self.inner.state.write().unwrap();
        state.writing_blocks.remove(&block.id());
        self.inner.metrics.storage_block_engine_block_writing.decrease(1);
        let inserted = state.evictable_blocks.insert(block.id());
        self.inner.metrics.storage_block_engine_block_evictable.increase(1);

        assert!(inserted);

        // Temporarily take pickers to make borrow checker happy.
        let mut pickers = std::mem::take(&mut state.eviction_pickers);

        // Notify pickers.
        for picker in pickers.iter_mut() {
            picker.on_block_evictable(
                EvictionInfo {
                    blocks: &self.inner.blocks,
                    evictable: &state.evictable_blocks,
                    clean: state.clean_blocks.len(),
                },
                block.id(),
            );
        }

        // Restore taken pickers after operations.

        std::mem::swap(&mut state.eviction_pickers, &mut pickers);
        assert!(pickers.is_empty());

        tracing::debug!(
            id = block.id(),
            "[block manager]: Block state transfers from writing to evictable."
        );

        let generation = self.inner.generations[block.id() as usize].load(Ordering::Relaxed);
        let reclaiming = self.reclaim_if_needed(&mut state);
        drop(state);
        self.inner.observer.emit(StorageEvent::BlockStateChanged {
            block: block.id(),
            generation,
            state: BlockState::Evictable,
        });
        if let Some(reclaiming) = reclaiming {
            self.start_reclaim(reclaiming);
        }
    }

    fn on_reclaim_finish(&self, block: Block) {
        let mut state = self.inner.state.write().unwrap();
        let id = block.id();
        state.reclaiming_blocks.remove(&block.id());
        self.inner.metrics.storage_block_engine_block_reclaiming.decrease(1);
        let waiter = state.clean_block_waiters.pop();
        let (generation, block_state) = if waiter.is_some() {
            let generation = self.inner.generations[id as usize].fetch_add(1, Ordering::Relaxed) + 1;
            state.writing_blocks.insert(id);
            self.inner.metrics.storage_block_engine_block_writing.increase(1);
            (generation, BlockState::Writing)
        } else {
            self.inner.metrics.storage_block_engine_block_clean.increase(1);
            state.clean_blocks.push_back(id);
            (
                self.inner.generations[id as usize].load(Ordering::Relaxed),
                BlockState::Clean,
            )
        };
        let reclaiming = self.reclaim_if_needed(&mut state);
        if state.reclaiming_blocks.is_empty() {
            for tx in std::mem::take(&mut state.reclaim_waiters) {
                let _ = tx.send(());
            }
        }
        drop(state);
        self.inner.observer.emit(StorageEvent::BlockStateChanged {
            block: id,
            generation,
            state: block_state,
        });
        if let Some(waiter) = waiter {
            let _ = waiter.send(block);
        }
        if let Some(reclaiming) = reclaiming {
            self.start_reclaim(reclaiming);
        }
    }

    fn reclaim_if_needed<'a>(&self, state: &mut RwLockWriteGuard<'a, State>) -> Option<ReclaimingBlock> {
        if state.clean_blocks.len() < self.inner.clean_block_threshold
            && state.reclaiming_blocks.len() < self.inner.reclaim_concurrency
        {
            if let Some(block) = self.evict(state) {
                state.reclaiming_blocks.insert(block.id());
                self.inner.metrics.storage_block_engine_block_reclaiming.increase(1);
                return Some(ReclaimingBlock {
                    block_manager: self.clone(),
                    block,
                    reinsert: true,
                });
            }
        }
        None
    }

    fn start_reclaim(&self, block: ReclaimingBlock) {
        let id = block.id();
        let generation = self.inner.generations[id as usize].load(Ordering::Relaxed);
        self.inner.observer.emit(StorageEvent::BlockStateChanged {
            block: id,
            generation,
            state: BlockState::Reclaiming,
        });
        let future = self.inner.reclaimer.reclaim(block);
        self.inner.spawner.spawn(future);
    }

    fn evict<'a>(&self, state: &mut RwLockWriteGuard<'a, State>) -> Option<Block> {
        let mut picked = None;

        if state.evictable_blocks.is_empty() {
            return None;
        }

        // Temporarily take pickers to make borrow checker happy.
        let mut pickers = std::mem::take(&mut state.eviction_pickers);

        // Pick a block to evict with pickers.
        for picker in pickers.iter_mut() {
            if let Some(block) = picker.pick(EvictionInfo {
                blocks: &self.inner.blocks,
                evictable: &state.evictable_blocks,
                clean: state.clean_blocks.len(),
            }) {
                picked = Some(block);
                break;
            }
        }

        // If no block is selected, just randomly pick one.
        let picked = picked.unwrap_or_else(|| state.evictable_blocks.iter().choose(&mut rand::rng()).copied().unwrap());

        // Update evictable map.
        let removed = state.evictable_blocks.remove(&picked);
        self.inner.metrics.storage_block_engine_block_evictable.decrease(1);
        assert!(removed);

        // Notify pickers.
        for picker in pickers.iter_mut() {
            picker.on_block_evict(
                EvictionInfo {
                    blocks: &self.inner.blocks,
                    evictable: &state.evictable_blocks,
                    clean: state.clean_blocks.len(),
                },
                picked,
            );
        }

        // Restore taken pickers after operations.
        std::mem::swap(&mut state.eviction_pickers, &mut pickers);
        assert!(pickers.is_empty());

        let block = self.inner.blocks[picked as usize].clone();
        tracing::debug!("[block manager]: Block {picked} is evicted.");

        Some(block)
    }

    pub fn wait_reclaim(&self) -> BoxFuture<'static, ()> {
        let mut state = self.inner.state.write().unwrap();
        if state.reclaiming_blocks.is_empty() {
            return ready(()).boxed();
        }
        let (tx, rx) = oneshot::channel();
        state.reclaim_waiters.push(tx);
        async move {
            let _ = rx.await;
        }
        .boxed()
    }
}

pub struct ReclaimingBlock {
    block_manager: BlockManager,
    block: Block,
    reinsert: bool,
}

impl ReclaimingBlock {
    pub(crate) fn reinsert(&self) -> bool {
        self.reinsert
    }
}

impl Deref for ReclaimingBlock {
    type Target = Block;

    fn deref(&self) -> &Self::Target {
        &self.block
    }
}

impl DerefMut for ReclaimingBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.block
    }
}

impl Drop for ReclaimingBlock {
    fn drop(&mut self) {
        self.block_manager.on_reclaim_finish(self.block.clone());
    }
}
