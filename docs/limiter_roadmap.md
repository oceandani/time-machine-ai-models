 Global Limiter Development Roadmap and Contract

Based on the principles of high availability and the Dependency Inversion Principle, we will adopt a four-step approach to integrate the GlobalLimiter.

📍 Development Steps

[Completed] Step A: Draft and Contract Definition

Produce the core/limiter.py skeleton and custom exception classes.

Establish the context management semantics of async with GlobalLimiter.acquire().

[Next] Step B: Business Layer Dependency Contract (Refactor Service)

Modify services/vlm_service.py.

Integrate the acquire semantics with Exponential Backoff.

Implement the defensive programming standard: "Release the Slot first, then execute sleep/wait".

[To Do] Step C: Refine Implementation Details and Test

Conduct Concurrency Testing for core/limiter.py.

Ensure asyncio.wait_for can accurately intercept timeouts within FastAPI's Uvicorn Workers.

[Future] Step D: Phase 2 Redis Replacement (Commercialization)

Develop RedisLimiter, strictly aligning with all contracts defined in Step A.

Achieve seamless switching; the business layer (vlm_service.py) code will not require a single line of modification.

🤝 Interface Contract

Any implementation of GlobalLimiter (including the future Redis version) must strictly guarantee the following contracts:

Single Entry Point: Must be encapsulated via @asynccontextmanager; the business logic side can only write async with.

Timeout Interruption: Must support the queue_wait_timeout parameter. If queueing times out, a QueueTimeoutError must be raised.

Mandatory Release: Regardless of whether the yielded business logic encounters an HTTP error, JSON parsing error, or is forcibly cancelled (CancelledError), the finally block must guarantee the lock is released.

Stateless Dependency: Cannot bind specific Job IDs or Request states inside the Limiter. The Limiter only recognizes the number of available Slots, not individual requests or users.

🛑 Business Layer Development Guidelines (For Step B)

In the upcoming Step B, vlm_service.py must strictly adhere to the following anti-deadlock rules:

Strictly Prohibited: Calling asyncio.sleep(retry_delay) inside the async with block.

Retry Strategy: When encountering a 429 error and preparing to retry, you must first return or break out of the async with block. Only after the lock has been safely released can you perform the sleep (Exponential Backoff).