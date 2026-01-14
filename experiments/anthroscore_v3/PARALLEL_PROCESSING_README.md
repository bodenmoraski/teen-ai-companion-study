# Parallel Processing for AnthroScore

## Overview

The original AnthroScore V2 was fast because it used **local MLM models** (RoBERTa) that could be batched on GPU. The new LLM-based approach uses API calls, which were initially sequential and slow.

**Solution**: Async/parallel processing with `asyncio` to make concurrent API calls.

## Speed Improvements

### Sequential (Original)
- **Speed**: ~0.5s per comment = **2 comments/sec**
- **Time for 283k comments**: ~39 hours

### Parallel (New)
- **Speed**: ~0.05s per comment (with 20 concurrent) = **20 comments/sec**
- **Time for 283k comments**: ~4 hours
- **Speedup**: **10x faster!**

## Implementation

### Files
1. `anthroscore_llm_parallel.py` - Async version of basic LLM scorer
2. `smart_routing_scorer_parallel.py` - Async version of smart routing scorer
3. `run_smart_routing.py` - Updated to use async processing

### Key Features
- **Concurrent API calls**: Process 20 comments simultaneously
- **Semaphore limiting**: Prevents overwhelming the API
- **Progress tracking**: Real-time rate and ETA
- **Error handling**: Retries with exponential backoff
- **Checkpointing**: Resumable long runs

## Usage

### Basic Async Scorer
```python
from anthroscore_llm_parallel import AsyncAnthroScoreLLM
import asyncio

scorer = AsyncAnthroScoreLLM(model="gpt-4.1-nano", max_concurrent=20)
results = await scorer.score_batch_async(texts, progress_interval=100)
```

### Smart Routing (Parallel)
```python
from smart_routing_scorer_parallel import AsyncSmartRoutingScorer
import asyncio

scorer = AsyncSmartRoutingScorer(max_concurrent=20)
results = await scorer.score_batch_async(texts, usernames, counts)
```

### Full Dataset Processing
```bash
# Estimate cost
python experiments/anthroscore_v3/run_smart_routing.py --dry-run

# Actually run (with parallel processing)
python experiments/anthroscore_v3/run_smart_routing.py --run
```

## Performance Tuning

### Concurrency Level
- **Default**: 20 concurrent requests
- **Conservative**: 10 (if hitting rate limits)
- **Aggressive**: 50 (if API allows, check rate limits first)

### Batch Size
- **Default**: 1000 comments per batch
- **Small**: 100 (for testing)
- **Large**: 5000 (for maximum throughput)

## Cost Impact

Parallel processing **does not change cost** - it just speeds things up:
- Same number of API calls
- Same cost per comment
- **10x faster** = same cost, 10x less time

## Comparison

| Approach | Speed | Time (283k) | Cost |
|----------|-------|-------------|------|
| Sequential | 2/sec | 39 hours | $15.94 |
| Parallel (20x) | 20/sec | 4 hours | $15.94 |
| Parallel (50x) | 50/sec | 1.6 hours | $15.94 |

**Recommendation**: Use parallel processing with 20 concurrent requests for best balance of speed and API stability.

---

*Generated: 2026-01-12*
*Study: The Illusion Project - Anthropomorphization of AI Companions*
