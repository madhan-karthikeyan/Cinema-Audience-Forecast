from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.monitoring.logging import get_logger
from app.monitoring.metrics import MODEL_LOADED

logger = get_logger(__name__)

LIGHTGBM_AVAILABLE: bool = False
XGBOOST_AVAILABLE: bool = False
CATBOOST_AVAILABLE: bool = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ModelVersion:
    name: str
    version: str
    path: Path
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    created_at: str = ""
    checksum: str = ""
    active: bool = False


class ModelRegistry:
    def __init__(self, base_path: Path = Path("models")):
        self.base_path = base_path
        self._versions: dict[str, list[ModelVersion]] = {}
        self._active: dict[str, ModelVersion] = {}
        self._loaded_models: dict[str, Any] = {}
        self._load_manifest()
        self._availability: dict[str, bool] = {
            "lightgbm": LIGHTGBM_AVAILABLE,
            "xgboost": XGBOOST_AVAILABLE,
            "catboost": CATBOOST_AVAILABLE,
        }

    def register(
        self,
        name: str,
        version: str,
        path: Path,
        metrics: dict,
        params: dict,
    ) -> ModelVersion:
        full_path = self.base_path / name / path
        if not full_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_path}")

        checksum = self._compute_checksum(full_path)
        mv = ModelVersion(
            name=name,
            version=version,
            path=full_path,
            metrics=metrics,
            params=params,
            created_at=__import__("datetime").datetime.utcnow().isoformat() + "Z",
            checksum=checksum,
            active=False,
        )
        self._versions.setdefault(name, []).append(mv)
        self._save_manifest()
        logger.info("model_registered", name=name, version=version, checksum=checksum)
        return mv

    def activate(self, name: str, version: str) -> None:
        for mv in self._versions.get(name, []):
            mv.active = False
            if mv.version == version:
                mv.active = True
                self._active[name] = mv
        self._save_manifest()
        logger.info("model_activated", name=name, version=version)

    def get_active(self, name: str) -> Optional[ModelVersion]:
        return self._active.get(name)

    def get_version(self, name: str, version: str) -> Optional[ModelVersion]:
        for mv in self._versions.get(name, []):
            if mv.version == version:
                return mv
        return None

    def list_versions(self, name: str) -> list[ModelVersion]:
        return self._versions.get(name, [])

    def list_models(self) -> list[str]:
        return list(self._versions.keys())

    def is_available(self, name: str) -> bool:
        return self._availability.get(name, False)

    def load_model(self, name: str) -> Optional[Any]:
        if name in self._loaded_models:
            return self._loaded_models[name]

        mv = self.get_active(name)
        if mv is None:
            logger.warning("no_active_version", model=name)
            return None

        model_path = mv.path
        if not model_path.exists():
            logger.warning("model_file_not_found", model=name, path=str(model_path))
            return None

        if not self._availability.get(name, False):
            logger.warning(
                "model_library_not_installed",
                model=name,
                library={
                    "lightgbm": "lightgbm",
                    "xgboost": "xgboost",
                    "catboost": "catboost",
                }.get(name, name),
            )
            return None

        try:
            model = self._do_load(name, model_path)
            self._loaded_models[name] = model
            MODEL_LOADED.labels(
                model_name=name, model_version=mv.version
            ).set(1)
            logger.info(
                "model_loaded",
                model=name,
                version=mv.version,
                path=str(model_path),
            )
            return model
        except Exception as e:
            logger.error(
                "model_load_failed",
                model=name,
                error=str(e),
                exc_info=True,
            )
            MODEL_LOADED.labels(model_name=name, model_version=mv.version).set(0)
            return None

    def unload_model(self, name: str) -> None:
        self._loaded_models.pop(name, None)
        mv = self.get_active(name)
        MODEL_LOADED.labels(
            model_name=name,
            model_version=mv.version if mv else "unknown",
        ).set(0)
        logger.info("model_unloaded", model=name)

    def load_all_models(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name in self.list_models():
            model = self.load_model(name)
            results[name] = model is not None
        loaded = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info("models_load_complete", loaded=loaded, total=total)
        return results

    def get_loaded_model(self, name: str) -> Optional[Any]:
        return self._loaded_models.get(name)

    @property
    def loaded_count(self) -> int:
        return len(self._loaded_models)

    @property
    def expected_count(self) -> int:
        return len(self._active)

    def _do_load(self, name: str, path: Path) -> Any:
        if name == "lightgbm":
            import lightgbm as lgb

            return lgb.Booster(model_file=str(path))

        if name == "xgboost":
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(str(path))
            return model

        if name == "catboost":
            from catboost import CatBoostRegressor

            model = CatBoostRegressor()
            model.load_model(str(path))
            return model

        raise ValueError(f"Unknown model type: {name}")

    def _compute_checksum(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_manifest(self) -> None:
        manifest_path = self.base_path / "registry.json"
        if not manifest_path.exists():
            logger.info("no_registry_manifest_found", path=str(manifest_path))
            return
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            for entry in data.get("models", []):
                mv = ModelVersion(
                    name=entry["name"],
                    version=entry["version"],
                    path=Path(entry["path"]),
                    metrics=entry.get("metrics", {}),
                    params=entry.get("params", {}),
                    created_at=entry.get("created_at", ""),
                    checksum=entry.get("checksum", ""),
                    active=entry.get("active", False),
                )
                self._versions.setdefault(mv.name, []).append(mv)
                if mv.active:
                    self._active[mv.name] = mv
            logger.info(
                "registry_manifest_loaded",
                model_count=len(data.get("models", [])),
            )
        except Exception as e:
            logger.error("registry_manifest_load_failed", error=str(e))

    def _save_manifest(self) -> None:
        manifest_path = self.base_path / "registry.json"
        models = []
        for _name, versions in self._versions.items():
            for mv in versions:
                models.append({
                    "name": mv.name,
                    "version": mv.version,
                    "path": str(mv.path),
                    "metrics": mv.metrics,
                    "params": mv.params,
                    "created_at": mv.created_at,
                    "checksum": mv.checksum,
                    "active": mv.active,
                })
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({"models": models}, f, indent=2)
