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
    cell::UnsafeCell,
    fmt::Debug,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use bitflags::bitflags;

use crate::eviction::Eviction;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Flags: u64 {
        const IN_INDEXER = 0b00000001;
        const IN_EVICTION = 0b00000010;
    }
}

pub struct Data<E>
where
    E: Eviction,
{
    pub key: E::Key,
    pub value: E::Value,
    pub properties: E::Properties,
    pub hash: u64,
    pub weight: usize,
}

/// [`Record`] holds the information of the cached entry.
pub struct Record<E>
where
    E: Eviction,
{
    data: Data<E>,
    state: UnsafeCell<E::State>,
    residency_sequence: u64,
    inserted_at_unix_micros: u64,
    last_accessed_at_unix_micros: AtomicU64,
    last_access_report_bucket: AtomicU64,
    /// Reference count used in the in-memory cache.
    refs: AtomicUsize,
    flags: AtomicU64,
}

unsafe impl<E> Send for Record<E> where E: Eviction {}
unsafe impl<E> Sync for Record<E> where E: Eviction {}

impl<E> Debug for Record<E>
where
    E: Eviction,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Record").field("hash", &self.data.hash).finish()
    }
}

impl<E> Record<E>
where
    E: Eviction,
{
    /// `state` field memory layout offset of the [`Record`].
    pub const STATE_OFFSET: usize = std::mem::offset_of!(Self, state);

    /// Create a record with data and no assigned cache residency sequence.
    pub fn new(data: Data<E>) -> Self {
        Self::new_with_residency_sequence(data, 0)
    }

    /// Create a record with data and a process-local cache residency sequence.
    pub(crate) fn new_with_residency_sequence(data: Data<E>, residency_sequence: u64) -> Self {
        let now = unix_micros();
        Record {
            data,
            state: Default::default(),
            residency_sequence,
            inserted_at_unix_micros: now,
            last_accessed_at_unix_micros: AtomicU64::new(now),
            last_access_report_bucket: AtomicU64::new(0),
            refs: AtomicUsize::new(0),
            flags: AtomicU64::new(0),
        }
    }

    /// Get the immutable reference of the record key.
    pub fn key(&self) -> &E::Key {
        &self.data.key
    }

    /// Get the immutable reference of the record value.
    pub fn value(&self) -> &E::Value {
        &self.data.value
    }

    /// Get the immutable reference of the record properties.
    pub fn properties(&self) -> &E::Properties {
        &self.data.properties
    }

    /// Get the record hash.
    pub fn hash(&self) -> u64 {
        self.data.hash
    }

    /// Get the record weight.
    pub fn weight(&self) -> usize {
        self.data.weight
    }

    /// Get the process-local sequence for this memory residency.
    pub fn residency_sequence(&self) -> u64 {
        self.residency_sequence
    }

    /// Get the wall-clock time when the record was created.
    pub fn inserted_at_unix_micros(&self) -> u64 {
        self.inserted_at_unix_micros
    }

    /// Get the wall-clock time when the record was last accessed.
    pub fn last_accessed_at_unix_micros(&self) -> u64 {
        self.last_accessed_at_unix_micros.load(Ordering::Relaxed)
    }

    /// Update the wall-clock time when the record was last accessed.
    pub fn touch_access_time(&self) -> u64 {
        let now = unix_micros();
        self.last_accessed_at_unix_micros.store(now, Ordering::Relaxed);
        now
    }

    /// Advance the reported access bucket, returning whether an event should be emitted.
    pub fn advance_access_report_bucket(&self, bucket: u64) -> bool {
        self.last_access_report_bucket
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (bucket > current).then_some(bucket)
            })
            .is_ok()
    }

    /// Get the record state wrapped with [`UnsafeCell`].
    ///
    /// # Safety
    pub fn state(&self) -> &UnsafeCell<E::State> {
        &self.state
    }

    /// Set in eviction flag with relaxed memory order.
    pub fn set_in_eviction(&self, val: bool) {
        self.set_flags(Flags::IN_EVICTION, val, Ordering::Release);
    }

    /// Get in eviction flag with relaxed memory order.
    pub fn is_in_eviction(&self) -> bool {
        self.get_flags(Flags::IN_EVICTION, Ordering::Acquire)
    }

    /// Set in indexer flag with relaxed memory order.
    pub fn set_in_indexer(&self, val: bool) {
        self.set_flags(Flags::IN_INDEXER, val, Ordering::Release);
    }

    /// Get in indexer flag with relaxed memory order.
    pub fn is_in_indexer(&self) -> bool {
        self.get_flags(Flags::IN_INDEXER, Ordering::Acquire)
    }

    /// Set the record atomic flags.
    pub fn set_flags(&self, flags: Flags, val: bool, order: Ordering) {
        match val {
            true => self.flags.fetch_or(flags.bits(), order),
            false => self.flags.fetch_and(!flags.bits(), order),
        };
    }

    /// Get the record atomic flags.
    pub fn get_flags(&self, flags: Flags, order: Ordering) -> bool {
        self.flags.load(order) & flags.bits() == flags.bits()
    }

    /// Get the atomic reference count.
    pub fn refs(&self) -> usize {
        self.refs.load(Ordering::Acquire)
    }

    /// Increase the atomic reference count.
    ///
    /// This function returns the new reference count after the op.
    pub fn inc_refs(&self, val: usize) -> usize {
        let old = self.refs.fetch_add(val, Ordering::SeqCst);
        tracing::trace!(
            "[record]: inc record (hash: {}) refs: {} => {}",
            self.hash(),
            old,
            old + val
        );
        old + val
    }

    /// Decrease the atomic reference count.
    ///
    /// This function returns the new reference count after the op.
    pub fn dec_refs(&self, val: usize) -> usize {
        let old = self.refs.fetch_sub(val, Ordering::SeqCst);
        tracing::trace!(
            "[record]: dec record (hash: {}) refs: {} => {}",
            self.hash(),
            old,
            old - val
        );
        old - val
    }
}

fn unix_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
        .try_into()
        .unwrap_or(u64::MAX)
}
