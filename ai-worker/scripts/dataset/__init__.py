"""Phase 9 dataset synthesis pipeline.

Scripts under this subpackage build the enterprise-document corpus and
stratified evaluation datasets used by Phase 10 (Optuna) and Phase 11
(README numbers). Entry points:

    python -m scripts.dataset.build_corpus
    python -m scripts.dataset.generate_queries
    python -m scripts.dataset.validate_dataset
    python -m scripts.dataset.generate_hard_set
    python -m scripts.dataset.synthesize_ocr_pages
    python -m scripts.dataset.generate_multimodal
    python -m scripts.dataset.generate_routing_cases
    python -m scripts.dataset.sample_check

See ai-worker/eval/datasets/README.md for the regeneration guide.
"""
