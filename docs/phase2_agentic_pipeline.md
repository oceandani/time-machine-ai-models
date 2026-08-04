🏗️ Phase 2: Agentic VLM & Self-Correction Pipeline (Architecture & Guidelines)

This document outlines the core development guidelines for Phase 2 of the Time Machine project, heavily based on high stability, debuggability, and MLOps best practices.

1. Core Development Guidelines

We will strictly adhere to the following 6 principles during development:

Strict Reproducibility: Every request sent to ComfyUI must explicitly define and log critical parameters like Seed, Prompt template version, Steps, CFG, and Denoise strength. Black-box randomness is strictly prohibited.

Comprehensive Artifacts Tracking: Each generation task must create an isolated job_id directory. Every iteration (Pass) must physically save: the source image, the preview Mask, the output image, and a JSON log detailing the parameters and VLM (Vision Language Model) decisions.

Resource Shielding: Implement strict Job Timeouts (e.g., 120s per Pass) and utilize asyncio.Semaphore at the FastAPI level to control concurrency, preventing GPU OOM (Out of Memory) crashes or Queue deadlocks.

Image Standardization: All input images must be pre-processed (Resized & Padded) by the Backend to ensure dimensions are multiples of 64 before hitting ComfyUI. This prevents dimension mismatch disasters during Inpainting.

Mask Visibility: All masks generated during the self-correction process must be exported and saved in a visual format (.png). This allows engineers to rapidly debug Inpainting failures visually.

Batch Testing for QA: Build CLI tools to allow batch inference against a fixed test dataset. This provides quantitative metrics on the VLM's Gate Pass Rate and average correction attempts.

2. Agentic Pipeline Flow (State Machine)

A complete "Time Machine Generation Task" is modeled as an asynchronous State Machine.

State Flow

[START] -> [Pre-processing] -> [Pass 0: Base Generation] -> [VLM QA Evaluation]
|
+-------------------------------------------------------------+
|
v
[Pass / Max Retries Reached?]
-> YES -> [Return Final Image & Cleanup] -> [END]
-> NO  -> [Generate Error Mask] -> [Pass N: Inpaint Correction] -> (Return to VLM QA Evaluation)

Artifacts Directory Structure (Example)

During runtime, the structure under the logs/jobs/ directory must reflect the following:

logs/jobs/job_abc123/
├── input_raw.jpg                # Original user upload
├── input_processed.jpg          # Standardized image (padded to multiple of 64)
├── pass0/
│   ├── prompt.json              # Exact payload sent to ComfyUI
│   ├── output.png               # Initial generated Base Image
│   └── qa_result.json           # VLM verdict (e.g., pass: false, reason: "Modern car")
├── pass1/
│   ├── error_mask.png           # Visualized bounding box mask from VLM
│   ├── prompt.json              # Inpaint parameters
│   ├── output.png               # Image after Inpaint correction
│   └── qa_result.json           # Second VLM verdict (e.g., pass: true)
└── final_result.png             # Final image cropped and returned to user


3. Action Items

To implement this architecture, we will build out the following modules:

Infrastructure: Create JobManager to handle atomic writes, directory scaffolding, and metadata logging.

Image Processor: Implement Resize, Padding, and Mask generation logic to enforce the 64-pixel multiple rule.

VLM Integrator: Integrate Gemini 2.0 Flash API with robust Prompt Engineering to output strictly validated JSON (Pass/Fail + Bounding Box coordinates).

Agent Loop: Assemble the above modules into services/agent_service.py, implementing the while loop and backoff retry logic.