# contracts/schema.py
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

AllowedMLClass = Literal[
    "Cell", "Cell-Multi", "Hot-Spot", "Hot-Spot-Multi",
    "Diode", "Diode-Multi", "Cracking", "Offline-Module",
    "Soiling", "Shadowing", "Vegetation", "No-Anomaly"
]

AllowedAgentCategory = Literal[
    "hotspot", "bypass_diode_failure", "cracking",
    "offline_module", "soiling", "shadowing", "vegetation", "none"
]

class SiteInfo(BaseModel):
    name: str
    sector: str
    array: str

class ImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

class DetectionItem(BaseModel):
    id: str = Field(min_length=1)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    bbox: list[int] = Field(min_length=4, max_length=4)
    polygon: Optional[list[list[int]]] = None  # Optional 4-corner [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] for angled drone panels
    hot_cell_bbox: Optional[list[int]] = None
    ml_class: AllowedMLClass
    agent_category: AllowedAgentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    top3: list[tuple[AllowedMLClass, float]] = Field(min_length=3, max_length=3)
    max_temp_c: float
    delta_t_c: Optional[float] = None
    flag_reason: Literal["model", "temperature", "model_and_temperature"]

    @model_validator(mode="after")
    def check_consistency(self):
        if self.agent_category == "none" and self.flag_reason != "temperature":
            raise ValueError("If agent_category is 'none', flag_reason must be 'temperature'")
        if self.agent_category != "none" and self.flag_reason == "temperature":
            raise ValueError("Conflict: faulty class cannot have 'temperature' alone.")
        return self

class SceneSummary(BaseModel):
    n_modules: int = Field(gt=0)
    n_predicted_anomalies: int = Field(ge=0)
    n_temperature_alerts: int = Field(ge=0)

class SceneDetectionsContract(BaseModel):
    schema_version: Literal["1.0"]
    scene_id: str = Field(min_length=1)
    site: SiteInfo
    image_size: ImageSize
    temperature_source: Literal["synthetic", "radiometric", "estimated_visual"]
    baseline_method: Literal["scene_module_median"]
    scene_baseline_temp_c: Optional[float] = None
    detections: list[DetectionItem]
    summary: SceneSummary