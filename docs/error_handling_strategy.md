🛡️ VLM Error Handling & Graceful Degradation Strategy

Based on the high availability requirements of commercial applications, we cannot allow the inherent instability of the VLM (Vision Language Model) to crash the main image generation pipeline (ComfyUI). Therefore, we strictly categorize VLM evaluation results and system errors into a tiered degradation strategy.

🚦 Three-Tier Response Mechanism

The VLM service response states are categorized into the following three tiers:

1. Tier 1: CONTENT_REJECT (Non-compliant Content) -> Trigger Self-Correction

Trigger Condition: The VLM connects successfully and explicitly detects modern artifacts (e.g., modern cars, LED streetlights) in the generated historical image.

System Action: pass = False. The Agent enters the standard Inpaint correction loop.

User Impact: The user waits slightly longer for the correction process but ultimately receives a high-quality, era-accurate image.

2. Tier 2: SYSTEM_WARNING_SKIP (Non-fatal System Exception) -> Graceful Degradation (Bypass)

Trigger Condition: The VLM encounters transient network fluctuations, continuous 429 Rate Limit errors, 503 Service Unavailable, or recurring JSON decoding failures (hallucinations).

System Action: pass = False (technically), but flagged with error_level = "SYSTEM_WARNING_SKIP". The Agent will abort the VQA inspection and directly return the best available Base Image to the user.

User Impact: The user still receives a Time Machine image (albeit occasionally containing minor anachronistic flaws). They will never encounter a white screen or a hard error message, ensuring baseline commercial conversion rates and UX.

3. Tier 3: SYSTEM_FATAL (Fatal System Error) -> Terminate Task

Trigger Condition: Severe configuration errors, such as a completely missing GEMINI_API_KEY in the .env file, or a 403 Forbidden error due to account suspension/unpaid bills.

System Action: Raises a RuntimeError. The FastAPI backend intercepts this and returns a standard HTTP 500 error to the frontend.

User Impact: The generation task explicitly fails. These errors are meant to be caught by monitoring systems during development or deployment and should theoretically never occur in a live production environment.