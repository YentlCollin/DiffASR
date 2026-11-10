"""Model-independent building blocks for MawJais discrete ASR diffusion."""

from .alignment import align_denoiser_logits, both_denoiser_alignments
from .attention import (
    build_prefix_canvas_allowed_mask,
    build_prefix_canvas_attention_mask,
    build_prefix_canvas_position_ids,
)
from .canvas import (
    CanvasChoice,
    CanvasEncoding,
    CanvasOverflowError,
    DurationBucketSelector,
    build_clean_canvas,
    pad_canvas_batch,
    truncate_at_first_turn_end,
)
from .checkpointing import (
    DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    load_diffusion_component_state,
    read_diffusion_checkpoint_manifest,
    save_diffusion_component_state,
    write_diffusion_checkpoint_manifest,
)
from .config import (
    CanvasConfig,
    ConfidenceSamplerConfig,
    DiffusionConfig,
    ResolvedObjective,
    resolve_objective,
)
from .conditional import (
    CONDITIONAL_ASR_VIEWS,
    CONDITIONAL_ASR_VIEW_BY_NAME,
    ConditionalASRView,
    conditional_asr_view,
    conditional_prompt_tag,
    conditional_view_from_sample,
    view_id_from_requests,
)
from .conditional_evaluation import (
    CONDITIONAL_EVALUATION_SCHEMA_VERSION,
    score_conditional_predictions,
    validate_paired_prediction_rows,
)
from .collator import (
    DiffusionCollatorCounters,
    MawJaisDiffusionCollator,
    NO_TRANSCRIPT_TARGET,
    NoisedCanvasBatch,
    build_mawjais_asr_prefix,
    corrupt_diffusion_batch,
    target_prompt_tag,
)
from .experiment_config import load_experiment_config, validate_experiment_config
from .isct import ISCTSecondStageBatch, build_isct_second_stage
from .noise_process import (
    DiscreteNoiseProcess,
    MaskNoiseProcess,
    UniformNoiseProcess,
    sample_diffusion_times,
)
from .model import (
    DiffusionForwardOutput,
    MawJaisDiffusionASRModel,
    pack_prefix_canvas_embeddings,
)
from .loading import (
    MawJaisRepositoryClasses,
    load_mawjais_r2_for_diffusion,
    load_repository_classes,
)
from .metrics import NOISE_BINS, denoising_token_metrics, ratios_from_additive_metrics
from .objectives import diffusion_loss, mdm_absorbing, x0_ce_all
from .sampler import (
    sample_diffusion,
    confidence_schedule_sample,
    mask_forbidden_logits,
    select_entropy_bounded_positions,
)
from .time_embedding import SinusoidalTimeFeatures, TimeConditioningMLP
from .token_policy import (
    DiffusionTokenPolicy,
    MawJaisTokenContract,
    build_diffusion_token_policy,
    hash_token_ids,
    validate_mawjais_token_contract,
)

__all__ = [
    "sample_diffusion",
    "CanvasChoice",
    "CanvasConfig",
    "CanvasEncoding",
    "CanvasOverflowError",
    "ConfidenceSamplerConfig",
    "CONDITIONAL_ASR_VIEWS",
    "CONDITIONAL_ASR_VIEW_BY_NAME",
    "CONDITIONAL_EVALUATION_SCHEMA_VERSION",
    "DIFFUSION_CHECKPOINT_SCHEMA_VERSION",
    "DiffusionConfig",
    "DiffusionCollatorCounters",
    "DiscreteNoiseProcess",
    "DiffusionForwardOutput",
    "DiffusionTokenPolicy",
    "ConditionalASRView",
    "DurationBucketSelector",
    "ISCTSecondStageBatch",
    "MaskNoiseProcess",
    "MawJaisDiffusionASRModel",
    "MawJaisDiffusionCollator",
    "MawJaisTokenContract",
    "MawJaisRepositoryClasses",
    "NO_TRANSCRIPT_TARGET",
    "NOISE_BINS",
    "NoisedCanvasBatch",
    "ResolvedObjective",
    "SinusoidalTimeFeatures",
    "TimeConditioningMLP",
    "UniformNoiseProcess",
    "align_denoiser_logits",
    "both_denoiser_alignments",
    "build_clean_canvas",
    "build_mawjais_asr_prefix",
    "build_prefix_canvas_allowed_mask",
    "build_prefix_canvas_attention_mask",
    "build_prefix_canvas_position_ids",
    "build_diffusion_token_policy",
    "build_isct_second_stage",
    "confidence_schedule_sample",
    "conditional_asr_view",
    "conditional_prompt_tag",
    "conditional_view_from_sample",
    "corrupt_diffusion_batch",
    "diffusion_loss",
    "denoising_token_metrics",
    "mask_forbidden_logits",
    "hash_token_ids",
    "validate_mawjais_token_contract",
    "load_mawjais_r2_for_diffusion",
    "load_diffusion_component_state",
    "load_repository_classes",
    "mdm_absorbing",
    "pad_canvas_batch",
    "pack_prefix_canvas_embeddings",
    "resolve_objective",
    "read_diffusion_checkpoint_manifest",
    "ratios_from_additive_metrics",
    "load_experiment_config",
    "sample_diffusion_times",
    "save_diffusion_component_state",
    "select_entropy_bounded_positions",
    "score_conditional_predictions",
    "truncate_at_first_turn_end",
    "target_prompt_tag",
    "x0_ce_all",
    "validate_experiment_config",
    "validate_paired_prediction_rows",
    "view_id_from_requests",
    "write_diffusion_checkpoint_manifest",
]
