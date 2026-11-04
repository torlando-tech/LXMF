import RNS
import RNS.Stamps

import os
import time
import math
import itertools
import multiprocessing
import inspect
import threading

from RNS.Stamps import stamp_workblock
from RNS.Stamps import stamp_valid
from RNS.Stamps import stamp_value

# LXMF-specific stamp parameters
WORKBLOCK_EXPAND_ROUNDS         = 3000
WORKBLOCK_EXPAND_ROUNDS_PN      = 1000
WORKBLOCK_EXPAND_ROUNDS_PEERING = 25
STAMP_SIZE                      = RNS.Identity.HASHLENGTH//8
PN_VALIDATION_POOL_MIN_SIZE     = 256

# The registry also reserves IDs while an ordinary upstream job is starting,
# so registering a native generator cannot replace an in-flight Python job.
active_jobs = {}
active_jobs_lock = threading.RLock()
external_generator = None
external_generator_cancel = None
external_generator_uses_token = False
external_generator_generation = 0


class ExternalStampCancellationToken:
    """Thread-safe cancellation state for cooperative native stamp generators."""

    def __init__(self, message_id):
        self.message_id = message_id
        self._event = threading.Event()

    @property
    def cancelled(self):
        return self._event.is_set()

    def is_cancelled(self):
        return self._event.is_set()

    def wait(self, timeout=None):
        return self._event.wait(timeout)


class ExternalStampJob:
    def __init__(self, message_id, cancel_callback):
        self.token = ExternalStampCancellationToken(message_id)
        self.cancel_callback = cancel_callback
        self._cancel_callback_called = False

    def request_cancel(self):
        # Called with active_jobs_lock held; callbacks are invoked after release.
        self.token._event.set()
        if self.cancel_callback is not None and not self._cancel_callback_called:
            self._cancel_callback_called = True
            return self.cancel_callback
        return None

    def invoke_cancel_callback(self, callback):
        if callback is not None:
            try:
                callback(self.token)
            except Exception as e:
                RNS.log(f"External stamp cancellation callback failed: {e}", RNS.LOG_ERROR)
                RNS.trace_exception(e)


def _accepts_cancellation_token(generator):
    try:
        inspect.signature(generator).bind(None, None, None)
        return True
    except (TypeError, ValueError):
        return False


def set_external_generator(generator, cancellation_callback=None, pass_cancellation_token=None):
    """Register ``generator(workblock, cost[, token])`` and optional cancellation.

    The cost is passed unchanged, including fractional costs. Opaque bridge
    callables can explicitly select token passing when inspection is unavailable.
    Replacing or clearing registration cancels all outstanding external jobs.
    """
    global external_generator, external_generator_cancel, external_generator_uses_token, external_generator_generation

    if generator is None and cancellation_callback is not None:
        raise ValueError("A cancellation callback requires an external generator")
    if generator is not None and not callable(generator):
        raise TypeError("External stamp generator must be callable")
    if cancellation_callback is not None and not callable(cancellation_callback):
        raise TypeError("External stamp cancellation callback must be callable")
    if pass_cancellation_token is not None and not isinstance(pass_cancellation_token, bool):
        raise TypeError("pass_cancellation_token must be a boolean or None")

    uses_token = False
    if generator is not None:
        uses_token = _accepts_cancellation_token(generator) if pass_cancellation_token is None else pass_cancellation_token

    with active_jobs_lock:
        external_generator_generation += 1
        external_generator = generator
        external_generator_cancel = cancellation_callback
        external_generator_uses_token = uses_token
        callbacks = [(job, job.request_cancel()) for job in active_jobs.values()
                     if isinstance(job, ExternalStampJob)]

    for job, callback in callbacks:
        job.invoke_cancel_callback(callback)


def _external_job_current(message_id, job, generator, generation):
    # Caller holds active_jobs_lock, including at the completion linearisation.
    return (not job.token.cancelled
            and generation == external_generator_generation
            and generator is external_generator
            and active_jobs.get(message_id) is job)


def generate_stamp(message_id, stamp_cost, expand_rounds=WORKBLOCK_EXPAND_ROUNDS, no_expand=False):
    with active_jobs_lock:
        generator = external_generator
        if message_id in active_jobs:
            if generator is None:
                raise RuntimeError(f"Duplicate stamp generation job for {RNS.prettyhexrep(message_id)}")
            RNS.log(f"Refusing duplicate external stamp job for {RNS.prettyhexrep(message_id)}", RNS.LOG_ERROR)
            return None, 0
        if generator is not None and message_id in RNS.Stamps.active_jobs:
            RNS.log(f"Refusing duplicate external stamp job for {RNS.prettyhexrep(message_id)}", RNS.LOG_ERROR)
            return None, 0
        generation = external_generator_generation
        uses_token = external_generator_uses_token
        job = ExternalStampJob(message_id, external_generator_cancel) if generator is not None else object()
        active_jobs[message_id] = job

    try:
        if generator is None:
            return RNS.Stamps.generate_stamp(message_id, stamp_cost, expand_rounds=expand_rounds, no_expand=no_expand, size=STAMP_SIZE)

        if stamp_cost < 0 or stamp_cost > RNS.Transport.STAMP_COST_CAP:
            raise ValueError("Invalid stamp cost, cannot generate")
        workblock = stamp_workblock(message_id, expand_rounds=expand_rounds, no_expand=no_expand)
        with active_jobs_lock:
            if not _external_job_current(message_id, job, generator, generation):
                return None, 0

        result = generator(workblock, stamp_cost, job.token) if uses_token else generator(workblock, stamp_cost)

        # Cancellation sentinels are not malformed payloads. Check before any
        # validation, then again when accepting a result to cover replacement.
        with active_jobs_lock:
            if not _external_job_current(message_id, job, generator, generation):
                return None, 0

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("result must be a (stamp, rounds) tuple")
        stamp, rounds = result
        if not isinstance(stamp, bytes):
            raise TypeError("stamp must be bytes")
        if len(stamp) != STAMP_SIZE:
            raise ValueError(f"stamp must be exactly {STAMP_SIZE} bytes")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
            raise TypeError("rounds must be a non-negative integer")
        if rounds > 0x7FFFFFFFFFFFFFFF:
            raise ValueError("rounds exceeds the supported 64-bit work counter")
        if not stamp_valid(stamp, stamp_cost, workblock):
            raise ValueError(f"stamp does not satisfy required cost {stamp_cost}")
        value = stamp_value(workblock, stamp)

        with active_jobs_lock:
            if not _external_job_current(message_id, job, generator, generation):
                return None, 0
            active_jobs.pop(message_id)
            return stamp, value

    except Exception as e:
        if generator is None:
            raise
        RNS.log(f"External stamp generator failed validation or execution: {e}", RNS.LOG_ERROR)
        RNS.trace_exception(e)
        return None, 0

    finally:
        with active_jobs_lock:
            if active_jobs.get(message_id) is job:
                active_jobs.pop(message_id)


def cancel_work(message_id):
    with active_jobs_lock:
        job = active_jobs.get(message_id)
        if isinstance(job, ExternalStampJob):
            callback = job.request_cancel()
        else:
            job = None

    if job is not None:
        job.invoke_cancel_callback(callback)
    else:
        return RNS.Stamps.cancel_work(message_id)

def validate_peering_key(peering_id, peering_key, target_cost):
    workblock = stamp_workblock(peering_id, expand_rounds=WORKBLOCK_EXPAND_ROUNDS_PEERING)
    if not stamp_valid(peering_key, target_cost, workblock): return False
    else: return True

def validate_pn_stamp(transient_data, target_cost):
    from .LXMessage import LXMessage
    if len(transient_data) <= LXMessage.LXMF_OVERHEAD+STAMP_SIZE: return None, None, None, None
    else:
        lxm_data     = transient_data[:-STAMP_SIZE]
        stamp        = transient_data[-STAMP_SIZE:]
        transient_id = RNS.Identity.full_hash(lxm_data)
        workblock    = stamp_workblock(transient_id, expand_rounds=WORKBLOCK_EXPAND_ROUNDS_PN)
        
        if not stamp_valid(stamp, target_cost, workblock): return None, None, None, None
        else:
            value = stamp_value(workblock, stamp)
            return transient_id, lxm_data, value, stamp

def validate_pn_stamps_job_simple(transient_list, target_cost):
    validated_messages = []
    for transient_data in transient_list:
        transient_id, lxm_data, value, stamp_data = validate_pn_stamp(transient_data, target_cost)
        if transient_id: validated_messages.append([transient_id, lxm_data, value, stamp_data])

    return validated_messages

def validate_pn_stamps_job_multip(transient_list, target_cost):
    cores      = multiprocessing.cpu_count()
    pool_count = min(cores, math.ceil(len(transient_list) / PN_VALIDATION_POOL_MIN_SIZE))
        
    RNS.log(f"Validating {len(transient_list)} stamps using {pool_count} processes...", RNS.LOG_VERBOSE)
    with multiprocessing.get_context("spawn").Pool(pool_count) as p:
        validated_entries = p.starmap(validate_pn_stamp, zip(transient_list, itertools.repeat(target_cost)))
    
    RNS.log(f"Validation pool completed for {len(transient_list)} stamps", RNS.LOG_VERBOSE)

    return [e for e in validated_entries if e[0] != None]

def validate_pn_stamps(transient_list, target_cost):
    non_mp_platform = RNS.vendor.platformutils.is_android()
    if len(transient_list) <= PN_VALIDATION_POOL_MIN_SIZE or non_mp_platform: return validate_pn_stamps_job_simple(transient_list, target_cost)
    else:                                                                     return validate_pn_stamps_job_multip(transient_list, target_cost)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        RNS.log("No cost argument provided", RNS.LOG_ERROR)
        exit(1)
    else:
        try: cost = int(sys.argv[1])
        except Exception as e:
            RNS.log(f"Invalid cost argument provided: {e}", RNS.LOG_ERROR)
            exit(1)

        rounds = 1
        if len(sys.argv) > 2:
            try: rounds = int(sys.argv[2])
            except Exception as e:
                RNS.log(f"Invalid rounds argument provided: {e}", RNS.LOG_ERROR)
                exit(1)

    RNS.loglevel = RNS.LOG_DEBUG
    
    for i in range(rounds):
        RNS.log("", RNS.LOG_DEBUG)
        RNS.log(f"Round {i+1} of {rounds}", RNS.LOG_DEBUG)

        RNS.log("Testing LXMF stamp generation", RNS.LOG_DEBUG)
        message_id = os.urandom(32)
        generate_stamp(message_id, cost)

        RNS.log("", RNS.LOG_DEBUG)
        RNS.log("Testing propagation stamp generation", RNS.LOG_DEBUG)
        message_id = os.urandom(32)
        generate_stamp(message_id, cost, expand_rounds=WORKBLOCK_EXPAND_ROUNDS_PN)

        RNS.log("", RNS.LOG_DEBUG)
        RNS.log("Testing peering key generation", RNS.LOG_DEBUG)
        message_id = os.urandom(32)
        generate_stamp(message_id, cost, expand_rounds=WORKBLOCK_EXPAND_ROUNDS_PEERING)

        RNS.log("", RNS.LOG_DEBUG)
        RNS.log("Testing non-expanded workblock generation", RNS.LOG_DEBUG)
        message_id = os.urandom(32)
        generate_stamp(message_id, cost, expand_rounds=0)

        # transient_list = []
        # st = time.time(); count = 10000
        # for i in range(count): transient_list.append(os.urandom(256))
        # validate_pn_stamps(transient_list, 5)
        # dt = time.time()-st; mps = count/dt
        # RNS.log(f"Validated {count} PN stamps in {RNS.prettytime(dt)}, {round(mps,1)} m/s", RNS.LOG_DEBUG)
