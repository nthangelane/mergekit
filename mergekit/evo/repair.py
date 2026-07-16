import gc
import itertools
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers


@dataclass
class RepairResult:
    child_model: torch.nn.Module
    repaired: bool
    probe_slope: float
    steps_used: int
    initial_loss: float
    probe_end_loss: float
    final_loss: float
    stopped_for_plateau: bool = False

    def metadata(self) -> Dict[str, Any]:
        return {
            "repaired": self.repaired,
            "probe_slope": self.probe_slope,
            "steps_used": self.steps_used,
            "initial_loss": self.initial_loss,
            "probe_end_loss": self.probe_end_loss,
            "final_loss": self.final_loss,
            "stopped_for_plateau": self.stopped_for_plateau,
        }


def _normalized_weights(weights: Optional[Sequence[float]], count: int) -> torch.Tensor:
    if weights is None or len(weights) != count:
        return torch.full((count,), 1.0 / float(count), dtype=torch.float32)
    values = torch.as_tensor(weights, dtype=torch.float32).abs()
    total = float(values.sum().item())
    if total <= 0 or not torch.isfinite(values).all():
        return torch.full((count,), 1.0 / float(count), dtype=torch.float32)
    return values / total


def _align_vocab_logits(logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Align padded or extended parent vocabularies to the child's output size."""
    parent_vocab_size = int(logits.shape[-1])
    if parent_vocab_size == vocab_size:
        return logits
    if parent_vocab_size > vocab_size:
        return logits[..., :vocab_size]

    padding = logits.new_full(
        (*logits.shape[:-1], vocab_size - parent_vocab_size),
        torch.finfo(logits.dtype).min,
    )
    return torch.cat((logits, padding), dim=-1)


def _isolated_cpu_rng(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        seed = int(kwargs.get("seed", 0))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return function(*args, **kwargs)

    return wrapped


class ParentDistiller:
    """CPU fp32 ensemble teacher whose output is a weighted probability mixture."""

    def __init__(
        self,
        parents: Sequence[Union[str, torch.nn.Module]],
        *,
        weights: Optional[Sequence[float]] = None,
        tau_distill: float = 2.0,
        tokenizer=None,
        trust_remote_code: bool = False,
    ):
        if not parents:
            raise ValueError("ParentDistiller requires at least one parent")
        if tau_distill <= 0:
            raise ValueError("tau_distill must be > 0")
        self.tau_distill = float(tau_distill)
        self.models: List[torch.nn.Module] = []
        parent_names = [str(parent) for parent in parents if isinstance(parent, str)]
        for parent in parents:
            if isinstance(parent, torch.nn.Module):
                model = parent
            else:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    parent,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    trust_remote_code=trust_remote_code,
                )
            model.to(device="cpu", dtype=torch.float32)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self.models.append(model)
        self.weights = _normalized_weights(weights, len(self.models))
        self.tokenizer = tokenizer
        if self.tokenizer is None and parent_names:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                parent_names[0], trust_remote_code=trust_remote_code
            )
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def logits(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        weights: Optional[Sequence[float]] = None,
        vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        mixture_weights = (
            self.weights
            if weights is None
            else _normalized_weights(weights, len(self.models))
        )
        mixture = None
        model_inputs = {
            key: value.to("cpu")
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.no_grad():
            for weight, model in zip(mixture_weights, self.models):
                output = model(**model_inputs)
                parent_logits = output.logits.float()
                if vocab_size is not None:
                    parent_logits = _align_vocab_logits(parent_logits, vocab_size)
                probabilities = F.softmax(parent_logits / self.tau_distill, dim=-1)
                weighted = float(weight.item()) * probabilities
                if mixture is None:
                    mixture = weighted
                else:
                    if mixture.shape != weighted.shape:
                        raise ValueError(
                            "Parent models produced incompatible logit shapes"
                        )
                    mixture = mixture + weighted
        if mixture is None:  # pragma: no cover - constructor prevents this
            raise RuntimeError("Parent mixture is empty")
        return torch.log(mixture.clamp_min(1e-12))


class _ReplayItems:
    def __init__(self, source: Iterable[Any]):
        self._source = iter(source)
        self._cache: List[Any] = []
        self._replay: Optional[Iterator[Any]] = None

    def next(self) -> Any:
        if self._replay is not None:
            return next(self._replay)
        try:
            item = next(self._source)
            self._cache.append(item)
            return item
        except StopIteration:
            if not self._cache:
                raise ValueError("Repair corpus produced no usable examples")
            self._replay = itertools.cycle(self._cache)
            return next(self._replay)


def _training_batches(
    corpus_iter: Iterable[Any],
    *,
    tokenizer,
    batch_size: int,
    seq_len: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    source = _ReplayItems(corpus_iter)
    while True:
        item = source.next()
        if isinstance(item, str):
            if tokenizer is None:
                raise ValueError("A tokenizer is required for a text repair corpus")
            texts = [item]
            while len(texts) < batch_size:
                next_item = source.next()
                if not isinstance(next_item, str):
                    raise TypeError("Repair corpus cannot mix text and tensor batches")
                texts.append(next_item)
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=seq_len,
            )
            yield {key: value.to("cpu") for key, value in encoded.items()}
            continue

        if isinstance(item, torch.Tensor):
            input_ids = item.to(dtype=torch.long, device="cpu")
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids[:, :seq_len]
            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
            continue

        if isinstance(item, dict) and "input_ids" in item:
            batch = {
                key: torch.as_tensor(value, device="cpu")
                for key, value in item.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            if batch["input_ids"].ndim == 1:
                batch = {
                    key: value.unsqueeze(0) if value.ndim == 1 else value
                    for key, value in batch.items()
                }
            batch = {key: value[:, :seq_len] for key, value in batch.items()}
            batch["input_ids"] = batch["input_ids"].long()
            batch.setdefault("attention_mask", torch.ones_like(batch["input_ids"]))
            yield batch
            continue

        raise TypeError(f"Unsupported repair corpus item: {type(item).__name__}")


def _student_teacher_kl(
    child_model: torch.nn.Module,
    distiller: ParentDistiller,
    batch: Dict[str, torch.Tensor],
    *,
    weights: Optional[Sequence[float]],
) -> torch.Tensor:
    model_inputs = {
        key: value.to("cpu")
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    student_logits = child_model(**model_inputs).logits.float()
    student_log_probs = F.log_softmax(student_logits / distiller.tau_distill, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = distiller.logits(
        model_inputs,
        weights=weights,
        vocab_size=int(student_logits.shape[-1]),
    )
    token_kl = torch.sum(
        student_probs * (student_log_probs - teacher_log_probs), dim=-1
    )
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask.to(dtype=token_kl.dtype)
        loss = torch.sum(token_kl * mask) / mask.sum().clamp_min(1.0)
    else:
        loss = token_kl.mean()
    return loss * (distiller.tau_distill**2)


@_isolated_cpu_rng
def repair(
    child_model: torch.nn.Module,
    distiller: ParentDistiller,
    corpus_iter: Iterable[Any],
    *,
    probe_steps: int = 100,
    max_steps: int = 500,
    gate_min_slope: float = 1e-5,
    lr: float = 1e-5,
    batch_size: int = 4,
    seq_len: int = 256,
    plateau_patience: int = 50,
    weights: Optional[Sequence[float]] = None,
    tokenizer=None,
    seed: int = 0,
) -> RepairResult:
    if probe_steps <= 0 or max_steps < probe_steps:
        raise ValueError("Require 0 < probe_steps <= max_steps")
    if plateau_patience <= 0:
        raise ValueError("plateau_patience must be > 0")

    child_model.to(device="cpu", dtype=torch.float32)
    child_model.train()
    initial_parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in child_model.named_parameters()
    }
    trainable_parameters = [
        parameter for parameter in child_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Child model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
    batches = _training_batches(
        corpus_iter,
        tokenizer=tokenizer or distiller.tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    reference_batch = next(batches)
    with torch.no_grad():
        initial_loss = float(
            _student_teacher_kl(
                child_model, distiller, reference_batch, weights=weights
            ).item()
        )

    steps_used = 0
    best_training_loss = float("inf")
    plateau_steps = 0
    stopped_for_plateau = False

    def train_step(batch: Dict[str, torch.Tensor]) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = _student_teacher_kl(child_model, distiller, batch, weights=weights)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    for step in range(probe_steps):
        train_step(reference_batch if step == 0 else next(batches))
        steps_used += 1

    child_model.eval()
    with torch.no_grad():
        probe_end_loss = float(
            _student_teacher_kl(
                child_model, distiller, reference_batch, weights=weights
            ).item()
        )
    probe_slope = (initial_loss - probe_end_loss) / float(probe_steps)

    if not np.isfinite(probe_slope) or probe_slope <= gate_min_slope:
        with torch.no_grad():
            for name, parameter in child_model.named_parameters():
                parameter.copy_(initial_parameters[name])
        child_model.eval()
        return RepairResult(
            child_model=child_model,
            repaired=False,
            probe_slope=float(probe_slope),
            steps_used=steps_used,
            initial_loss=initial_loss,
            probe_end_loss=probe_end_loss,
            final_loss=initial_loss,
        )

    child_model.train()
    for _ in range(probe_steps, max_steps):
        training_loss = train_step(next(batches))
        steps_used += 1
        if training_loss < best_training_loss - 1e-12:
            best_training_loss = training_loss
            plateau_steps = 0
        else:
            plateau_steps += 1
            if plateau_steps >= plateau_patience:
                stopped_for_plateau = True
                break

    child_model.eval()
    with torch.no_grad():
        final_loss = float(
            _student_teacher_kl(
                child_model, distiller, reference_batch, weights=weights
            ).item()
        )
    return RepairResult(
        child_model=child_model,
        repaired=True,
        probe_slope=float(probe_slope),
        steps_used=steps_used,
        initial_loss=initial_loss,
        probe_end_loss=probe_end_loss,
        final_loss=final_loss,
        stopped_for_plateau=stopped_for_plateau,
    )


def load_repair_corpus(corpus: str, *, max_examples: int) -> List[str]:
    if corpus != "wikitext-train-slice":
        raise ValueError(f"Unsupported repair corpus: {corpus!r}")
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        streaming=True,
    )
    texts: List[str] = []
    for record in dataset:
        text = str(record.get("text") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= max_examples:
            break
    if not texts:
        raise ValueError("WikiText repair corpus yielded no non-empty examples")
    return texts


def blend_weights_for_genotype(genome, genotype: np.ndarray) -> List[float]:
    model_count = len(getattr(getattr(genome, "definition", None), "models", []) or [])
    if model_count <= 0:
        return []
    flat = np.asarray(genotype, dtype=np.float32).reshape(-1)
    values = None
    if hasattr(genome, "model_selection_dim") and hasattr(genome, "layer_group_dim"):
        layer_values = []
        for layer_index in range(int(getattr(genome, "num_layer_groups", 1))):
            start = layer_index * int(genome.layer_group_dim) + int(
                getattr(genome, "method_dim", 0)
            )
            layer_values.append(flat[start : start + model_count])
        values = np.mean(np.stack(layer_values), axis=0)
    else:
        template = genome.initial_genotype(random=False)
        shape = tuple(template.shape)
        if len(shape) == 4 and int(np.prod(shape)) == flat.size:
            structured = flat.reshape(shape)
            values = np.mean(np.abs(structured[:, :, :, 0]), axis=(0, 2))
    if values is None:
        return [1.0 / float(model_count)] * model_count
    normalized = _normalized_weights(values.tolist(), model_count)
    return [float(value) for value in normalized.tolist()]


def repair_checkpoint(
    checkpoint_path: str,
    distiller: ParentDistiller,
    corpus_iter: Iterable[Any],
    *,
    config,
    weights: Optional[Sequence[float]],
    seed: int,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    child_model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    result = repair(
        child_model,
        distiller,
        corpus_iter,
        probe_steps=config.probe_steps,
        max_steps=config.max_steps,
        gate_min_slope=config.gate_min_slope,
        lr=config.lr,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        plateau_patience=config.plateau_patience,
        weights=weights,
        seed=seed,
    )
    if result.repaired:
        result.child_model.save_pretrained(checkpoint_path, safe_serialization=True)
    metadata = result.metadata()
    del child_model
    gc.collect()
    return metadata
