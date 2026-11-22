import hashlib
import io
import json
import os
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from core.logging import logger

load_dotenv()

# ==============================================================================
# 1. 系統常數、絕對路徑解析與 main.py 啟動自檢相容全域變數 (零硬數值散落)
# ==============================================================================
VLM_TEMPERATURE = 0.1
VLM_PLANNER_TEMPERATURE = 0.1
VLM_AUDITOR_TEMPERATURE = 0.05

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
EVIDENCE_REGISTRY_VERSION = "v10.5_phase1_disjoint_isolated"
DEFAULT_EVIDENCE_SAMPLE_DIR = "/Users/danielee/Downloads/era project test sample"
EVIDENCE_REGISTRY_JSON_PATH = os.path.join(
    CONFIG_DIR, "evidence_registry.json"
)
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

HISTORICAL_FILENAME_MARKERS = (
    "1950",
    "1960",
    "1970",
    "1972",
    "1980",
    "五十年代",
    "六十年代",
    "七十年代",
    "八十年代",
    "old",
    "hist",
    "ref",
    "archive",
    "vintage",
    "舊",
    "歷史",
    "昔日",
)
MODERN_FILENAME_EXCLUSIONS = (
    "2010",
    "2015",
    "2020",
    "2024",
    "2025",
    "2026",
    "modern",
    "now",
    "temp_input",
    "現在",
    "今日",
)
MONOCHROME_SATURATION_THRESHOLD = 18.0
DAYLIGHT_LUMINANCE_THRESHOLD = 92.0
DEFAULT_NEUTRAL_LUMA = 128.0

DEFAULT_PRIMARY_ENV_MODEL = "gemini-flash-lite-latest"
FALLBACK_GEMINI_MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest")

ACTIVE_MODEL: Optional[str] = None
CANDIDATE_MODELS_QUEUE: List[str] = []
DEAD_MODELS_BLACKLIST: Set[str] = {
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
}

RATIO_MIN = 0.0
RATIO_MAX = 1.0
MIN_RATIO_SPAN = 0.04
LEGACY_COORD_SCALE = 1000
MIN_POLICY_MATRIX_KEYS = 4

VP_X_MIN_CLAMP = 0.28
VP_X_MAX_CLAMP = 0.72
VP_Y_MIN_CLAMP = 0.35
VP_Y_MAX_CLAMP = 0.68

ARCH_HEIGHT_MIN_CLAMP = 0.35
ARCH_HEIGHT_MAX_CLAMP = 0.90
LOW_RISE_MAX_HEIGHT_CAP = 0.44
DEFAULT_LOW_RISE_MAX_STORIES = 7
MAX_LOW_RISE_STORIES = DEFAULT_LOW_RISE_MAX_STORIES
DEFAULT_CANYON_MAX_STORIES = 45
DEFAULT_ESTIMATED_STORIES = 5
DEFAULT_FALLBACK_HEIGHT_RATIO = 0.50
DEFAULT_TALLEST_BUILDING_BBOX = (0.0, 0.5, 1.0, 0.9)

# 與 inference_service.py 及 global_rebuild_workflow_api.json 100% 同步之 IPAdapter 常數
DEFAULT_IPADAPTER_WEIGHT = 0.58
DEFAULT_IPADAPTER_END_AT = 0.82

# 單點透視無重疊包絡常數 (跨模組共用)
PERSPECTIVE_OUTER_ROOF_Y_RATIO = 0.42
PERSPECTIVE_INNER_CORRIDOR_X_LEFT = 0.34
PERSPECTIVE_INNER_CORRIDOR_X_RIGHT = 0.66
CORRIDOR_HALF_SPAN_X = 0.14
MIN_CORRIDOR_X_LEFT_BOUND = 0.22
MAX_CORRIDOR_X_LEFT_BOUND = 0.42
MIN_CORRIDOR_X_RIGHT_BOUND = 0.58
MAX_CORRIDOR_X_RIGHT_BOUND = 0.78

MIN_SKY_CUTOFF_Y_RATIO = 0.56
PERSPECTIVE_SKY_AREA_DISCOUNT = 0.40
DEFAULT_ROAD_TOP_Y_RATIO = 0.72

# 都會峽谷與公園摩天樓模式 (KEEP_CANYON_MASS) 無重疊互補幾何常數
CANYON_SKY_SIDE_Y_BOTTOM = 0.42
CANYON_CENTRAL_Y_TOP = 0.16
CANYON_TOWER_HALF_SPAN_X = 0.06
CANYON_MIN_ARCH_HEIGHT_RATIO = 0.85
CANYON_MIN_SKY_VISIBILITY_RATIO = 0.10

# 填海/水岸船塢模式 (DEMOLISH_TO_WATER_LEVEL) 無重疊分層常數
WATERFRONT_SKY_Y_BOTTOM = 0.24
WATERFRONT_RIDGE_Y_BOTTOM = 0.36
WATERFRONT_WATER_Y_TOP = 0.52
WATERFRONT_SHORE_X_SPAN = 0.30

DEFAULT_VANISHING_POINT_X = 0.48
DEFAULT_VANISHING_POINT_Y = 0.56
DEFAULT_HISTORICAL_CEILING_RATIO = 0.44
MIN_SKY_VISIBILITY_RATIO = 0.16
MAX_REQUIRED_SKY_VISIBILITY_CAP = 0.28
HEIGHT_ENVELOPE_TOLERANCE = 0.06
HUMAN_CONFIRM_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_CONTRACT_CONFIDENCE = 0.90
DEFAULT_LOCKED_ZONE_CONFIDENCE = 0.95
PROXY_MODE_CONFIDENCE_CAP = 0.72

MIN_PASSING_CLARITY_SCORE = 42.0
MAX_SCORE_ON_VETO = 80.0
PERCENTAGE_SCALE = 100.0
MIN_FOUNDATION_PASSING_ZONES = 1

LAYER_INDEX_MACRO_BACKDROP = 1
LAYER_INDEX_MESO_STRUCTURE = 2
LAYER_INDEX_MICRO_ENTITIES = 3

MAX_DYNAMIC_DISCOVERED_MODELS = 5

NEGATED_HAZE_PATTERNS = (
    r"\b(?:no|without|free of|zero)\s+(?:atmospheric\s+)?(?:haze|fog|mist|smog|blur)\b",
)
FORBIDDEN_DEGRADING_PATTERNS = (
    r"\bsubtle atmospheric haze\b",
    r"\batmospheric haze\b",
    r"\bslight atmospheric haze\b",
    r"\bwarm cinematic 1970s color grading\b",
    r"\bwarm color grading\b",
    r"\bvibrant analog film grain\b",
    r"\banalog film grain\b",
    r"\b1960s photographic grain\b",
    r"\bphotographic grain\b",
    r"\bfilm grain\b",
    r"\bmonochrome\b",
    r"\bblack and white\b",
    r"\bgreyscale\b",
    r"\bgrayscale\b",
    r"\bsmoggy\b",
    r"\bhazy\b",
    r"\bhaze\b",
    r"\bfoggy\b",
    r"\bfog\b",
    r"\bmisty\b",
    r"\bmist\b",
    r"\bdusty\b",
    r"\bsepia\b",
    r"\bblurry\b",
)
DAYLIGHT_FORBIDDEN_NIGHT_PATTERNS = (
    r"\bat dusk\b",
    r"\bdusk\b",
    r"\bnighttime\b",
    r"\bnight scene\b",
    r"\bnight\b",
    r"\bdark steel\b",
    r"\bdark building\b",
    r"\bglowing white incandescent bulb\b",
    r"\bglowing neon\b",
    r"\bwith headlights on\b",
    r"\bwith headlights\b",
    r"\bheadlights\b",
)

try:
    _api_key = os.getenv("GEMINI_API_KEY")
    client = (
        genai.Client(api_key=_api_key, http_options={"api_version": "v1beta"})
        if _api_key
        else genai.Client(http_options={"api_version": "v1beta"})
    )
except Exception as e:
    logger.error(f"🚨 Gemini SDK 初始化失敗，請檢查環境變數: {e}")
    client = None


async def get_candidate_models(api_client: Any = None) -> List[str]:
    primary_env_model = os.environ.get(
        "VLM_MODEL_NAME", DEFAULT_PRIMARY_ENV_MODEL
    )
    seed_candidates = [primary_env_model, *FALLBACK_GEMINI_MODELS]

    discovered_models: List[str] = []
    active_client = api_client or client
    if active_client is not None:
        try:
            for m in active_client.models.list():
                raw_name = getattr(m, "name", "") or ""
                short_name = (
                    raw_name.split("/")[-1] if "/" in raw_name else raw_name
                )
                actions = getattr(m, "supported_actions", None) or []
                if (
                    short_name.startswith("gemini-")
                    and "flash" in short_name
                    and short_name not in DEAD_MODELS_BLACKLIST
                    and "tts" not in short_name
                    and "image" not in short_name
                    and "audio" not in short_name
                    and (not actions or "generateContent" in actions)
                ):
                    discovered_models.append(short_name)
                    if len(discovered_models) >= MAX_DYNAMIC_DISCOVERED_MODELS:
                        break
        except Exception as e:
            logger.debug(f"動態查詢模型清單略過，使用預設候選佇列: {e}")

    ordered_queue: List[str] = []
    seen = set()
    for model_name in seed_candidates + discovered_models:
        if (
            model_name
            and model_name not in seen
            and model_name not in DEAD_MODELS_BLACKLIST
        ):
            seen.add(model_name)
            ordered_queue.append(model_name)

    return ordered_queue


# ==============================================================================
# 2. v10.5 全地貌四算子、目標基底與三模態枚舉定義
# ==============================================================================
class SceneChangeOperator(str, Enum):
    KEEP = "KEEP"
    DEMOLISH = "DEMOLISH"
    REPLACE = "REPLACE"
    GEOMORPHIC_CHANGE = "GEOMORPHIC_CHANGE"


class TargetHistoricalSubstrate(str, Enum):
    SUBSTRATE_SKY = "SUBSTRATE_SKY"
    SUBSTRATE_WATER_AND_DOCKYARD = "SUBSTRATE_WATER_AND_DOCKYARD"
    SUBSTRATE_STREET_CORRIDOR = "SUBSTRATE_STREET_CORRIDOR"
    SUBSTRATE_ARCHITECTURAL_FACADE = "SUBSTRATE_ARCHITECTURAL_FACADE"
    SUBSTRATE_NATURAL_TERRAIN = "SUBSTRATE_NATURAL_TERRAIN"


class PerspectivePolicy(str, Enum):
    LOCK_STREET_CORRIDOR = "LOCK_STREET_CORRIDOR"
    ALIGN_LANDMARK_VIEWPOINT = "ALIGN_LANDMARK_VIEWPOINT"
    REBUILD_FOREGROUND_PLANE = "REBUILD_FOREGROUND_PLANE"


class VerticalMassPolicy(str, Enum):
    KEEP_CANYON_MASS = "KEEP_CANYON_MASS"
    TRUNCATE_TO_LOW_RISE = "TRUNCATE_TO_LOW_RISE"
    DEMOLISH_TO_WATER_LEVEL = "DEMOLISH_TO_WATER_LEVEL"
    EXTRUDE_HISTORICAL_MASS = "EXTRUDE_HISTORICAL_MASS"


class SurfaceTopoPolicy(str, Enum):
    VERNACULAR_FACADE_SIGNAGE = "VERNACULAR_FACADE_SIGNAGE"
    RESTYLE_BILLBOARDS_FACADE = "RESTYLE_BILLBOARDS_FACADE"
    RECONSTRUCT_DOCKYARD_INFRA = "RECONSTRUCT_DOCKYARD_INFRA"


class BackdropSkyPolicy(str, Enum):
    RELEASE_UPPER_SKY = "RELEASE_UPPER_SKY"
    KEEP_EXISTING_SKY = "KEEP_EXISTING_SKY"
    LOCK_MOUNTAIN_RIDGELINE = "LOCK_MOUNTAIN_RIDGELINE"


class UrbanDensityClass(str, Enum):
    HIGH_DENSITY_METROPOLITAN_CANYON = "HIGH_DENSITY_METROPOLITAN_CANYON"
    DENSE_COMMERCIAL_CORRIDOR = "DENSE_COMMERCIAL_CORRIDOR"
    INDUSTRIAL_PORT_WATERFRONT = "INDUSTRIAL_PORT_WATERFRONT"
    LOW_DENSITY_SETTLEMENT = "LOW_DENSITY_SETTLEMENT"


class TopologicalZoneType(str, Enum):
    SKY_CANVAS = "SKY_CANVAS"
    STREET_CORRIDOR = "STREET_CORRIDOR"
    CENTRAL_BACKDROP = "CENTRAL_BACKDROP"
    LEFT_FACADE = "LEFT_FACADE"
    RIGHT_FACADE = "RIGHT_FACADE"
    WATER_AND_DOCKYARD = "WATER_AND_DOCKYARD"
    LOCKED_RIDGELINE = "LOCKED_RIDGELINE"


class RegionActionType(str, Enum):
    DETERMINISTIC_LOCK = "DETERMINISTIC_LOCK"
    GEOMETRY_GUIDED_RESTYLE = "GEOMETRY_GUIDED_RESTYLE"
    TRUNCATE_AND_REBUILD = "TRUNCATE_AND_REBUILD"
    TOPOGRAPHICAL_REPLACE = "TOPOGRAPHICAL_REPLACE"


class InpaintMaskMode(str, Enum):
    ZERO_LOCK = "ZERO_LOCK"
    FEATHERED_RESTYLE = "FEATHERED_RESTYLE"
    FULL_RECONSTRUCT = "FULL_RECONSTRUCT"


class ControlRepresentationType(str, Enum):
    ORIGINAL_DEPTH_GRADIENT = "ORIGINAL_DEPTH_GRADIENT"
    TRUNCATED_LOW_RISE_DEPTH = "TRUNCATED_LOW_RISE_DEPTH"
    FLATTENED_WATER_PLANE = "FLATTENED_WATER_PLANE"
    STRUCTURAL_EDGE_FRAME = "STRUCTURAL_EDGE_FRAME"


class PostCompositeRule(str, Enum):
    PIXEL_EXACT_COPY_BACK = "PIXEL_EXACT_COPY_BACK"
    BLEND_FEATHER_SEAM = "BLEND_FEATHER_SEAM"
    NONE = "NONE"


class MediaStyleClassification(str, Enum):
    AUTHENTIC_PHOTOGRAPH = "AUTHENTIC_PHOTOGRAPH"
    CARTOON_OR_ILLUSTRATION = "CARTOON_OR_ILLUSTRATION"
    VIDEO_GAME_3D_RENDER = "VIDEO_GAME_3D_RENDER"
    PAINTING_OR_POSTER = "PAINTING_OR_POSTER"


# ==============================================================================
# 3. Phase 1 全地貌史實證據卡 (矮樓街區、都會峽谷、公園摩天樓、填海水域)
# ==============================================================================
DEFAULT_PHASE1_EVIDENCE_CARDS: List[Dict[str, Any]] = [
    {
        "evidence_ref_id": "EVID_HK_CWB_1960S_01",
        "location_keywords": [
            "causeway bay",
            "銅鑼灣",
            "軒尼詩",
            "怡和",
            "yee wo",
            "hennessy",
            "sogo",
        ],
        "era_keywords": ["1950", "1960", "60s", "五十年代", "六十年代"],
        "archive_source_title": "HK Public Records Office & Gwulo 1960s Causeway Bay Hennessy Road / Yee Wo Street Color Archival Collection",
        "grounding_mode": "DIRECT_ARCHIVAL_EVIDENCE",
        "max_historical_stories": 6,
        "height_to_street_width_ratio": 0.44,
        "ipadapter_weight": DEFAULT_IPADAPTER_WEIGHT,
        "ipadapter_end_at": DEFAULT_IPADAPTER_END_AT,
        "required_policy_matrix": {
            "perspective_policy": "LOCK_STREET_CORRIDOR",
            "vertical_mass_policy": "TRUNCATE_TO_LOW_RISE",
            "surface_topo_policy": "VERNACULAR_FACADE_SIGNAGE",
            "backdrop_sky_policy": "RELEASE_UPPER_SKY",
        },
        "conjugate_landmarks": [
            "Hennessy Road / Yee Wo Street double-track tramway corridor axis",
            "Left and right 5-to-6 story Tong Lau perspective roofline envelope",
            "Central street corridor perspective vanishing point",
        ],
        "operator_mapping_rules": {
            "SKY_CANVAS": "DEMOLISH",
            "CENTRAL_BACKDROP": "REPLACE",
            "LEFT_FACADE": "REPLACE",
            "RIGHT_FACADE": "REPLACE",
            "STREET_CORRIDOR": "REPLACE",
        },
        "verified_color_palette_prompt": (
            "full-color 1960s Kodachrome daylight street photograph, warm sunlit ochre and beige concrete facades, "
            "vibrant vermilion-red, gold, and cobalt-blue vertical calligraphy shop signs, "
            "olive-green and cream double-decker tram, red and cream buses"
        ),
        "verified_architectural_features": (
            "low-rise 5-story 1960s British Colonial Hong Kong reinforced concrete cantilevered Tong Lau buildings "
            "with open balconies, horizontal window rows, ground-floor shopfront awnings, and projecting vertical Chinese signs"
        ),
        "verified_street_features": (
            "wide sunlit asphalt avenue with embedded steel tram tracks, 1960s olive-green wooden-body double-decker Hongkong Tramways tram, "
            "vintage red-and-cream CMB double-decker bus, classic black British Morris Oxford sedans, "
            "and Hong Kong pedestrians in authentic 1960s short-sleeve collared shirts, trousers, cheongsam, and summer dresses"
        ),
    },
    {
        "evidence_ref_id": "EVID_NYC_TSQ_1970S_01",
        "location_keywords": [
            "times square",
            "time square",
            "broadway",
            "時代廣場",
        ],
        "era_keywords": ["1960", "1970", "1972", "70s", "七十年代"],
        "archive_source_title": "NYC Municipal Archives 1972 Broadway & 7th Avenue Times Square Color Archival Survey",
        "grounding_mode": "DIRECT_ARCHIVAL_EVIDENCE",
        "max_historical_stories": DEFAULT_CANYON_MAX_STORIES,
        "height_to_street_width_ratio": 0.85,
        "ipadapter_weight": DEFAULT_IPADAPTER_WEIGHT,
        "ipadapter_end_at": DEFAULT_IPADAPTER_END_AT,
        "required_policy_matrix": {
            "perspective_policy": "LOCK_STREET_CORRIDOR",
            "vertical_mass_policy": "KEEP_CANYON_MASS",
            "surface_topo_policy": "RESTYLE_BILLBOARDS_FACADE",
            "backdrop_sky_policy": "KEEP_EXISTING_SKY",
        },
        "conjugate_landmarks": [
            "One Times Square central wedge building with SONY / Canadian Club / Coca-Cola stacked vintage signs",
            "Y-shaped Broadway and 7th Avenue road intersection",
            "Right-side brownstone masonry tower with GORDON'S / BOND / WOOLWORTH marquees",
        ],
        "operator_mapping_rules": {
            "SKY_CANVAS": "KEEP",
            "CENTRAL_BACKDROP": "REPLACE",
            "LEFT_FACADE": "REPLACE",
            "RIGHT_FACADE": "REPLACE",
            "STREET_CORRIDOR": "REPLACE",
        },
        "verified_color_palette_prompt": (
            "full-color 1972 Kodachrome archival street photograph, natural daylight illumination, warm ruby-red, "
            "amber-gold, and classic painted theatre marquees, brownstone and buff brick masonry facades, bright chrome-bumper yellow taxicabs"
        ),
        "verified_architectural_features": (
            "1970s New York Times Square masonry commercial buildings covered with large vintage hand-painted movie billboards, "
            "exposed steel-frame SONY and Coca-Cola signs on central wedge tower, and classic broadway theatre marquees"
        ),
        "verified_street_features": (
            "wide asphalt Broadway intersection in natural daylight, boxy 1970s American chrome-bumper yellow Checker Marathon taxicabs, "
            "vintage 1970s steel-body American sedans, and New York pedestrians in authentic 1970s flared trousers, trench coats, vintage jackets, and retro attire"
        ),
    },
    {
        "evidence_ref_id": "EVID_NYC_CENTRAL_PARK_1970S_01",
        "location_keywords": [
            "central park",
            "manhattan",
            "fifth avenue",
            "central park south",
            "中央公園",
            "曼克頓",
            "曼哈頓",
        ],
        "era_keywords": ["1960", "1970", "70s", "七十年代"],
        "archive_source_title": "NYC Parks & Municipal Archives 1970s Central Park & Midtown Manhattan Skyline Collection",
        "grounding_mode": "DIRECT_ARCHIVAL_EVIDENCE",
        "max_historical_stories": 55,
        "height_to_street_width_ratio": 0.85,
        "ipadapter_weight": DEFAULT_IPADAPTER_WEIGHT,
        "ipadapter_end_at": DEFAULT_IPADAPTER_END_AT,
        "required_policy_matrix": {
            "perspective_policy": "ALIGN_LANDMARK_VIEWPOINT",
            "vertical_mass_policy": "KEEP_CANYON_MASS",
            "surface_topo_policy": "VERNACULAR_FACADE_SIGNAGE",
            "backdrop_sky_policy": "KEEP_EXISTING_SKY",
        },
        "conjugate_landmarks": [
            "Midtown / Central Park South 1970s masonry and steel skyscraper skyline",
            "Central Park natural lawns, rock outcrops, and tree-lined walkways",
        ],
        "operator_mapping_rules": {
            "SKY_CANVAS": "KEEP",
            "CENTRAL_BACKDROP": "REPLACE",
            "LEFT_FACADE": "REPLACE",
            "RIGHT_FACADE": "REPLACE",
            "STREET_CORRIDOR": "REPLACE",
        },
        "verified_color_palette_prompt": (
            "full-color 1970s Kodachrome daylight photograph, lush olive and emerald park foliage, warm limestone, "
            "buff brick, and bronze-tinted 1970s Manhattan skyscrapers under natural daylight"
        ),
        "verified_architectural_features": (
            "1970s Manhattan skyline bordering Central Park featuring classic limestone Art Deco and mid-century International Style skyscrapers "
            "(Plaza Hotel, General Motors Building, Essex House style) without modern 2020s glass supertall needle towers"
        ),
        "verified_street_features": (
            "1970s Central Park pathways and park drives with vintage cast-iron lampposts, wood-slat park benches, "
            "and New Yorkers strolling in authentic 1970s casual clothing, bell-bottom jeans, and vintage knitwear"
        ),
    },
    {
        "evidence_ref_id": "EVID_HK_WATERFRONT_RECLAMATION_01",
        "location_keywords": [
            "taikoo",
            "quarry bay",
            "wan chai",
            "convention",
            "太古",
            "船塢",
            "鰂魚涌",
            "灣仔",
            "會展",
        ],
        "era_keywords": ["1950", "1960", "1970", "1980", "六十年代", "七十年代"],
        "archive_source_title": "HK Lands Department Historical Coastline & Taikoo Dockyard / Victoria Harbour Archives",
        "grounding_mode": "DIRECT_ARCHIVAL_EVIDENCE",
        "max_historical_stories": 6,
        "height_to_street_width_ratio": 0.42,
        "ipadapter_weight": DEFAULT_IPADAPTER_WEIGHT,
        "ipadapter_end_at": DEFAULT_IPADAPTER_END_AT,
        "required_policy_matrix": {
            "perspective_policy": "ALIGN_LANDMARK_VIEWPOINT",
            "vertical_mass_policy": "DEMOLISH_TO_WATER_LEVEL",
            "surface_topo_policy": "RECONSTRUCT_DOCKYARD_INFRA",
            "backdrop_sky_policy": "LOCK_MOUNTAIN_RIDGELINE",
        },
        "conjugate_landmarks": [
            "Distant natural mountain ridgeline (Mount Parker / Kowloon Ridgeline)",
            "Historical pre-reclamation stone seawall coastline boundary",
        ],
        "operator_mapping_rules": {
            "LOCKED_RIDGELINE": "KEEP",
            "SKY_CANVAS": "KEEP",
            "WATER_AND_DOCKYARD": "GEOMORPHIC_CHANGE",
            "CENTRAL_BACKDROP": "GEOMORPHIC_CHANGE",
            "LEFT_FACADE": "REPLACE",
            "RIGHT_FACADE": "REPLACE",
            "STREET_CORRIDOR": "GEOMORPHIC_CHANGE",
        },
        "verified_color_palette_prompt": (
            "full-color 1960s Kodachrome archival daylight photograph, weathered terracotta red-brick industrial workshops, "
            "oxide-red and slate-grey steel portal cranes, warm granite stone seawalls, deep greenish-blue Victoria Harbour water"
        ),
        "verified_architectural_features": (
            "1960s Taikoo Dockyard coastal industrial complex, red-brick machine shops, sloping concrete slipways, "
            "steel hammerhead cranes, and stone quay walls against green mountain foothills"
        ),
        "verified_street_features": (
            "open greenish-blue harbour water, dry dock basins, moored steel cargo steamships with red-and-black hulls, "
            "tugboats, and dockyard workers in 1960s industrial workwear along the masonry seawall"
        ),
    },
]


def _ensure_evidence_registry_on_disk() -> List[Dict[str, Any]]:
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        needs_write = True
        if os.path.exists(EVIDENCE_REGISTRY_JSON_PATH):
            with open(EVIDENCE_REGISTRY_JSON_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if existing_data.get("version") == EVIDENCE_REGISTRY_VERSION:
                needs_write = False
                return existing_data.get(
                    "evidence_cards", DEFAULT_PHASE1_EVIDENCE_CARDS
                )

        if needs_write:
            with open(EVIDENCE_REGISTRY_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": EVIDENCE_REGISTRY_VERSION,
                        "evidence_cards": DEFAULT_PHASE1_EVIDENCE_CARDS,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        return DEFAULT_PHASE1_EVIDENCE_CARDS
    except Exception as e:
        logger.warning(f"讀寫 evidence_registry.json 略過，使用內建證據集: {e}")
        return DEFAULT_PHASE1_EVIDENCE_CARDS


def _is_monochrome_image(image_bytes: bytes) -> bool:
    """檢測歷史參考照片是否為黑白灰階相片，以啟動結構與色彩解耦。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            hsv_img = img.convert("HSV")
            sat_channel = np.array(hsv_img)[:, :, 1]
            mean_sat = float(np.mean(sat_channel))
            return mean_sat < MONOCHROME_SATURATION_THRESHOLD
    except Exception:
        return False


def _detect_daylight_illumination(image_bytes: bytes) -> Tuple[float, bool]:
    """量測用戶輸入原圖之整體明度 (Luma)，若為日間照片則啟動日間光照守恆模式。"""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray_arr = np.array(img.convert("L"), dtype=np.float32)
            mean_luma = float(np.mean(gray_arr))
            return round(mean_luma, 1), (
                mean_luma >= DAYLIGHT_LUMINANCE_THRESHOLD
            )
    except Exception:
        return DEFAULT_NEUTRAL_LUMA, True


def retrieve_historical_evidence_context(
    location: str, year: str, input_image_bytes: bytes
) -> Dict[str, Any]:
    cards = _ensure_evidence_registry_on_disk()
    sample_dir = os.environ.get(
        "EVIDENCE_SAMPLE_DIR", DEFAULT_EVIDENCE_SAMPLE_DIR
    )
    input_sha256 = hashlib.sha256(input_image_bytes).hexdigest()
    input_luma, input_is_daylight = _detect_daylight_illumination(
        input_image_bytes
    )

    loc_lower = (location or "").lower()
    matched_card: Optional[Dict[str, Any]] = None
    for card in cards:
        loc_kws = card.get("location_keywords", [])
        if any(kw in loc_lower for kw in loc_kws):
            matched_card = card
            break

    if matched_card is None:
        matched_card = {
            "evidence_ref_id": f"PROXY_{re.sub(r'[^A-Z0-9]', '', location.upper())[:8]}_{re.sub(r'[^0-9]', '', year)[:4] or 'ERA'}",
            "location_keywords": [loc_lower],
            "archive_source_title": f"Spatial-Temporal Neighboring District Proxy for {location} ({year})",
            "grounding_mode": "SPATIAL_TEMPORAL_PROXY",
            "max_historical_stories": DEFAULT_LOW_RISE_MAX_STORIES,
            "height_to_street_width_ratio": DEFAULT_HISTORICAL_CEILING_RATIO,
            "ipadapter_weight": DEFAULT_IPADAPTER_WEIGHT,
            "ipadapter_end_at": DEFAULT_IPADAPTER_END_AT,
            "required_policy_matrix": {
                "perspective_policy": "LOCK_STREET_CORRIDOR",
                "vertical_mass_policy": "TRUNCATE_TO_LOW_RISE",
                "surface_topo_policy": "VERNACULAR_FACADE_SIGNAGE",
                "backdrop_sky_policy": "RELEASE_UPPER_SKY",
            },
            "conjugate_landmarks": [
                "Primary street corridor vanishing point",
                "Adjacent building setback baselines",
            ],
            "operator_mapping_rules": {
                "SKY_CANVAS": "DEMOLISH",
                "CENTRAL_BACKDROP": "REPLACE",
                "LEFT_FACADE": "REPLACE",
                "RIGHT_FACADE": "REPLACE",
                "STREET_CORRIDOR": "REPLACE",
            },
            "verified_color_palette_prompt": f"full-color {year} Kodachrome archival photograph, natural daylight, authentic period material colors",
            "verified_architectural_features": f"authentic {year} {location} regional mid-to-low-rise vernacular architecture matching neighboring street proportions",
            "verified_street_features": f"authentic {year} {location} period street surface, period vehicles, and pedestrians in authentic {year} attire",
        }

    card_loc_kws = [k.lower() for k in matched_card.get("location_keywords", [])]
    discovered_sample_files: List[str] = []
    matched_ref_image_path: Optional[str] = None
    matched_ref_image_bytes: Optional[bytes] = None
    fallback_color_ref_bytes: Optional[bytes] = None
    fallback_color_ref_name: Optional[str] = None

    if os.path.isdir(sample_dir):
        try:
            for fname in sorted(os.listdir(sample_dir)):
                if not fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    continue
                full_path = os.path.join(sample_dir, fname)
                if not os.path.isfile(full_path):
                    continue
                discovered_sample_files.append(fname)

                fname_lower = fname.lower()
                if any(ex in fname_lower for ex in MODERN_FILENAME_EXCLUSIONS):
                    continue
                has_hist_marker = any(
                    m in fname_lower for m in HISTORICAL_FILENAME_MARKERS
                )
                if not has_hist_marker:
                    continue

                with open(full_path, "rb") as img_f:
                    f_bytes = img_f.read()
                if hashlib.sha256(f_bytes).hexdigest() == input_sha256:
                    continue

                is_mono = _is_monochrome_image(f_bytes)
                if not is_mono and fallback_color_ref_bytes is None:
                    fallback_color_ref_bytes = f_bytes
                    fallback_color_ref_name = fname

                if (
                    any(lk in fname_lower for lk in card_loc_kws)
                    and matched_ref_image_bytes is None
                ):
                    matched_ref_image_path = full_path
                    matched_ref_image_bytes = f_bytes
        except Exception as e:
            logger.warning(f"掃描樣本目錄 {sample_dir} 時發生非致命異常: {e}")

    ref_is_monochrome = (
        _is_monochrome_image(matched_ref_image_bytes)
        if matched_ref_image_bytes is not None
        else False
    )
    color_ref_bytes = (
        matched_ref_image_bytes
        if (matched_ref_image_bytes is not None and not ref_is_monochrome)
        else fallback_color_ref_bytes
    )

    ref_label = (
        f"{os.path.basename(matched_ref_image_path)} ({'黑白結構參考+彩色色譜補償' if ref_is_monochrome else '全彩歷史參考圖'})"
        if matched_ref_image_path
        else f"結構化證據卡 + 懷舊彩色色譜 ({fallback_color_ref_name or '內建色票'})"
    )
    logger.info(
        f"📚 [Stage 1 歷史證據模態] 樣本庫: '{sample_dir}' (共 {len(discovered_sample_files)} 檔) | "
        f"證據卡: {matched_card['evidence_ref_id']} | 匹配參考源: {ref_label} | "
        f"原圖明度: {input_luma} ({'日間守恆模式' if input_is_daylight else '原圖低光模式'})"
    )

    return {
        "evidence_card": matched_card,
        "sample_dir": sample_dir,
        "discovered_sample_count": len(discovered_sample_files),
        "reference_image_path": matched_ref_image_path,
        "reference_image_bytes": matched_ref_image_bytes,
        "reference_is_monochrome": ref_is_monochrome,
        "color_reference_bytes": color_ref_bytes,
        "input_luma": input_luma,
        "input_is_daylight": input_is_daylight,
    }


# ==============================================================================
# 4. 向下相容啟動自檢 Schema (供 main.py 啟動時測試 Image+Schema 全能力使用)
# ==============================================================================
class GeographyProfile(BaseModel):
    background_type: str = Field(
        description="Must be one of: 'EXPANSIVE_SKY', 'MOUNTAIN_BACKDROP', 'HARBOUR_WATERLINE'"
    )
    current_building_height_ratio: float
    max_architecture_height_ratio: float
    ground_surface: str


class VisualArchetype(BaseModel):
    camera_angle: str
    architectural_prompt: str


class ProtectedZone(BaseModel):
    category: str
    confidence: float
    bbox: List[int]


class VLMResponse(BaseModel):
    schema_version: str = Field(default="v10.5_cot_geometric")
    reasoning_steps: List[str]
    pass_validation: bool
    error_level: str
    primary_archetype: str
    geography_profile: GeographyProfile
    visual_archetype: VisualArchetype
    protected_zones: List[ProtectedZone]


# ==============================================================================
# 5. v10.5 跨年代場景變化圖 & 三圖對照盲測取證 QA Schema
# ==============================================================================
class LandmarkCorrespondenceSchema(BaseModel):
    landmark_id: str
    landmark_description: str
    x_ratio: float
    y_ratio: float
    cross_era_stability_confidence: float


class PolicyMatrix4DSchema(BaseModel):
    perspective_policy: str
    vertical_mass_policy: str
    surface_topo_policy: str
    backdrop_sky_policy: str


class PlannerGeographySchema(BaseModel):
    background_type: str
    urban_density_class: str
    current_building_height_ratio: float
    max_architecture_height_ratio: float
    min_sky_visibility_ratio: float
    vanishing_point_x_ratio: float
    vanishing_point_y_ratio: float
    ground_surface: str


class RegionContractSchema(BaseModel):
    region_id: str
    layer_index: int
    topological_zone: str
    change_operator: str
    target_substrate: str
    anchor_landmark_id: str
    relative_topology_relation: str
    evidence_ref_id: str
    uncertainty_flag: bool
    uncertainty_reason: str
    semantic_class: str
    x_min_ratio: float
    y_min_ratio: float
    x_max_ratio: float
    y_max_ratio: float
    polygon_xy_ratios_flat: List[float]
    action_type: str
    historical_evidence_source: str
    regional_positive_prompt: str
    regional_negative_prompt: str
    confidence: float
    requires_human_confirmation: bool
    inpaint_mask_mode: str
    control_representation: str
    post_composite_rule: str


class ReconstructionPlanResponse(BaseModel):
    schema_version: str = Field(default="v10.5_scene_change_graph")
    reasoning_steps: List[str]
    already_historically_authentic: bool
    primary_archetype: str
    conjugate_landmarks: List[LandmarkCorrespondenceSchema]
    policy_matrix: PolicyMatrix4DSchema
    geography_profile: PlannerGeographySchema
    visual_archetype: VisualArchetype
    region_contracts: List[RegionContractSchema]


class VisualEvidenceInventory(BaseModel):
    media_style_classification: str = Field(
        description="Must be one of: 'AUTHENTIC_PHOTOGRAPH', 'CARTOON_OR_ILLUSTRATION', 'VIDEO_GAME_3D_RENDER', 'PAINTING_OR_POSTER'."
    )
    is_photorealistic_documentary_photo: bool
    is_full_color_not_monochrome: bool = Field(
        description="True ONLY IF the Candidate Generated Image is a full-color photograph. False if black-and-white, greyscale, or monochromatic."
    )
    daylight_illumination_preserved: bool = Field(
        description="True if Candidate Generated Image preserves natural daytime brightness when Modern Input is daytime."
    )
    tallest_building_bbox_xyxy: List[float]
    estimated_max_building_stories: int
    visible_sky_bbox_xyxy: List[float]
    primary_vehicle_bbox_xyxy: List[float]
    vehicles_structurally_realistic_and_on_lanes: bool
    pedestrians_attire_and_appearance_authentic: bool = Field(
        description="True ONLY IF visible pedestrians wear period-authentic clothing with natural human proportions (NOT modern 2020s clothing/backpacks/smartphones and NOT distorted/ghostly blobs)."
    )


class ZoneAuditItem(BaseModel):
    topological_zone: str
    zone_passed: bool
    observed_visual_content: str


class SceneIdentityAudit(BaseModel):
    pass_scene_identity: bool
    road_scale_and_width_preserved: bool
    regional_urban_identity_verified: bool
    no_village_or_alley_degradation: bool
    identity_failure_reason: str


class Track1VisualConsistencyAudit(BaseModel):
    pass_track1: bool
    perspective_and_corridor_aligned: bool
    envelope_and_sky_compliant: bool
    render_integrity_and_color_valid: bool
    measured_building_height_ratio: float
    measured_sky_visibility_ratio: float
    modern_artifacts_detected: List[str]
    track1_failure_reason: str


class Track2HistoricalEvidenceAudit(BaseModel):
    pass_track2: bool
    architectural_typology_verified: bool
    urban_density_and_commercial_fidelity: bool
    signage_and_vehicles_valid: bool
    verified_historical_features: List[str]
    historical_discrepancies: List[str]
    track2_failure_reason: str


class DualTrackQAResponse(BaseModel):
    schema_version: str = Field(default="v10.5_tri_modal_auditor")
    reasoning_steps: List[str]
    visual_evidence_inventory: VisualEvidenceInventory
    zone_audits: List[ZoneAuditItem]
    scene_identity_audit: SceneIdentityAudit
    track1_audit: Track1VisualConsistencyAudit
    track2_audit: Track2HistoricalEvidenceAudit
    overall_pass: bool
    primary_defect_zone: str
    failed_region_ids: List[str]
    failed_topological_zones: List[str]
    remediation_directives: List[str]


# ==============================================================================
# 6. 內部確定性工具與「全地貌 100% 無重疊鑲嵌分區」幾何正規化函數
# ==============================================================================
def _detect_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _clean_json_markdown(raw_text: str) -> str:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _strip_haze_and_degrading_terms(
    prompt_text: str, enforce_daylight: bool = True
) -> str:
    if not prompt_text:
        return ""
    cleaned = prompt_text
    cleaned = re.sub(
        r"\bno\s+crisp\s+clear\s+daylight\b",
        "crisp clear daylight",
        cleaned,
        flags=re.IGNORECASE,
    )
    for neg_pat in NEGATED_HAZE_PATTERNS:
        cleaned = re.sub(neg_pat, "clear daylight", cleaned, flags=re.IGNORECASE)
    for deg_pat in FORBIDDEN_DEGRADING_PATTERNS:
        cleaned = re.sub(deg_pat, "", cleaned, flags=re.IGNORECASE)
    if enforce_daylight:
        for night_pat in DAYLIGHT_FORBIDDEN_NIGHT_PATTERNS:
            cleaned = re.sub(
                night_pat, "natural daylight", cleaned, flags=re.IGNORECASE
            )
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,")


def _normalize_ratio(
    val: Any, default_val: float = DEFAULT_FALLBACK_HEIGHT_RATIO
) -> float:
    try:
        f_val = float(val)
        if f_val > RATIO_MAX:
            f_val = f_val / float(LEGACY_COORD_SCALE)
        return max(RATIO_MIN, min(RATIO_MAX, round(f_val, 4)))
    except (ValueError, TypeError):
        return default_val


def _sanitize_box_ratios(
    x_min: Any, y_min: Any, x_max: Any, y_max: Any
) -> Dict[str, float]:
    rx1 = _normalize_ratio(x_min, RATIO_MIN)
    ry1 = _normalize_ratio(y_min, RATIO_MIN)
    rx2 = _normalize_ratio(x_max, RATIO_MAX)
    ry2 = _normalize_ratio(y_max, RATIO_MAX)
    if rx1 > rx2:
        rx1, rx2 = rx2, rx1
    if ry1 > ry2:
        ry1, ry2 = ry2, ry1
    if (rx2 - rx1) < MIN_RATIO_SPAN:
        rx2 = min(RATIO_MAX, rx1 + MIN_RATIO_SPAN)
    if (ry2 - ry1) < MIN_RATIO_SPAN:
        ry2 = min(RATIO_MAX, ry1 + MIN_RATIO_SPAN)
    return {
        "x_min_ratio": rx1,
        "y_min_ratio": ry1,
        "x_max_ratio": rx2,
        "y_max_ratio": ry2,
    }


def _normalize_topological_zone(raw_zone: Any, sem_cls: str) -> str:
    valid_values = {e.value for e in TopologicalZoneType}
    z_str = str(raw_zone or "").strip().upper()
    if z_str in valid_values:
        return z_str
    s_upper = f"{z_str} {sem_cls}".upper()
    if "SKY" in s_upper:
        return TopologicalZoneType.SKY_CANVAS.value
    if (
        "STREET" in s_upper
        or "ROAD" in s_upper
        or "TRAFFIC" in s_upper
        or "VEHICLE" in s_upper
        or "WALKWAY" in s_upper
    ):
        return TopologicalZoneType.STREET_CORRIDOR.value
    if "LEFT" in s_upper:
        return TopologicalZoneType.LEFT_FACADE.value
    if "RIGHT" in s_upper:
        return TopologicalZoneType.RIGHT_FACADE.value
    if "WATER" in s_upper or "DOCK" in s_upper or "HARBOUR" in s_upper:
        return TopologicalZoneType.WATER_AND_DOCKYARD.value
    if "MOUNTAIN" in s_upper or "RIDGE" in s_upper:
        return TopologicalZoneType.LOCKED_RIDGELINE.value
    return TopologicalZoneType.CENTRAL_BACKDROP.value


def _derive_operator_and_substrate(
    topo_zone: str,
    raw_operator: Any,
    vertical_policy: str,
    backdrop_policy: str,
    evidence_card: Dict[str, Any],
) -> Tuple[str, str, str, str, str, str]:
    card_op_map = evidence_card.get("operator_mapping_rules", {})
    valid_ops = {e.value for e in SceneChangeOperator}
    op_str = str(raw_operator or "").strip().upper()
    if topo_zone in card_op_map:
        op_str = card_op_map[topo_zone]
    elif op_str not in valid_ops:
        if topo_zone == TopologicalZoneType.LOCKED_RIDGELINE.value:
            op_str = SceneChangeOperator.KEEP.value
        elif (
            topo_zone == TopologicalZoneType.SKY_CANVAS.value
            and backdrop_policy == BackdropSkyPolicy.KEEP_EXISTING_SKY.value
        ):
            op_str = SceneChangeOperator.KEEP.value
        elif (
            topo_zone == TopologicalZoneType.SKY_CANVAS.value
            and backdrop_policy == BackdropSkyPolicy.RELEASE_UPPER_SKY.value
        ):
            op_str = SceneChangeOperator.DEMOLISH.value
        elif vertical_policy == VerticalMassPolicy.DEMOLISH_TO_WATER_LEVEL.value:
            op_str = SceneChangeOperator.GEOMORPHIC_CHANGE.value
        else:
            op_str = SceneChangeOperator.REPLACE.value

    if op_str == SceneChangeOperator.KEEP.value:
        substrate = (
            TargetHistoricalSubstrate.SUBSTRATE_SKY.value
            if topo_zone == TopologicalZoneType.SKY_CANVAS.value
            else TargetHistoricalSubstrate.SUBSTRATE_NATURAL_TERRAIN.value
        )
        return (
            op_str,
            substrate,
            RegionActionType.DETERMINISTIC_LOCK.value,
            InpaintMaskMode.ZERO_LOCK.value,
            ControlRepresentationType.ORIGINAL_DEPTH_GRADIENT.value,
            PostCompositeRule.PIXEL_EXACT_COPY_BACK.value,
        )

    if op_str == SceneChangeOperator.DEMOLISH.value:
        return (
            op_str,
            TargetHistoricalSubstrate.SUBSTRATE_SKY.value,
            RegionActionType.TRUNCATE_AND_REBUILD.value,
            InpaintMaskMode.FULL_RECONSTRUCT.value,
            ControlRepresentationType.TRUNCATED_LOW_RISE_DEPTH.value,
            PostCompositeRule.BLEND_FEATHER_SEAM.value,
        )

    if op_str == SceneChangeOperator.GEOMORPHIC_CHANGE.value:
        return (
            op_str,
            TargetHistoricalSubstrate.SUBSTRATE_WATER_AND_DOCKYARD.value,
            RegionActionType.TOPOGRAPHICAL_REPLACE.value,
            InpaintMaskMode.FULL_RECONSTRUCT.value,
            ControlRepresentationType.FLATTENED_WATER_PLANE.value,
            PostCompositeRule.BLEND_FEATHER_SEAM.value,
        )

    substrate = (
        TargetHistoricalSubstrate.SUBSTRATE_STREET_CORRIDOR.value
        if topo_zone == TopologicalZoneType.STREET_CORRIDOR.value
        else TargetHistoricalSubstrate.SUBSTRATE_ARCHITECTURAL_FACADE.value
    )
    return (
        SceneChangeOperator.REPLACE.value,
        substrate,
        RegionActionType.GEOMETRY_GUIDED_RESTYLE.value,
        InpaintMaskMode.FEATHERED_RESTYLE.value,
        ControlRepresentationType.STRUCTURAL_EDGE_FRAME.value,
        PostCompositeRule.BLEND_FEATHER_SEAM.value,
    )


def _apply_polygon_and_sync_fields(
    contract: Dict[str, Any],
    poly_xy: List[List[float]],
) -> None:
    """由單一多邊形頂點清單自動派生 box_ratios、polygon_xy_ratios_flat 與 bbox_xyxy_1000，確保 100% 數學恆等。"""
    xs = [pt[0] for pt in poly_xy]
    ys = [pt[1] for pt in poly_xy]
    box_ratios = _sanitize_box_ratios(min(xs), min(ys), max(xs), max(ys))
    contract.update(box_ratios)
    contract["polygon_xy_ratios"] = [
        [round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in poly_xy
    ]
    contract["polygon_xy_ratios_flat"] = [
        round(float(coord), 4)
        for pt in contract["polygon_xy_ratios"]
        for coord in pt
    ]
    contract["vertical_envelope_ratio"] = [
        box_ratios["y_min_ratio"],
        box_ratios["y_max_ratio"],
    ]
    bbox_xyxy = [
        int(round(box_ratios["x_min_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box_ratios["y_min_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box_ratios["x_max_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box_ratios["y_max_ratio"] * LEGACY_COORD_SCALE)),
    ]
    contract["bbox_xyxy_1000"] = bbox_xyxy
    contract["bbox"] = bbox_xyxy


def _build_perspective_aligned_polygon(
    topo_zone: str,
    vertical_policy: str,
    enforce_sky_truncation: bool,
    sky_cutoff_y: float,
    vp_x: float,
    vp_y: float,
    box_ratios: Dict[str, float],
    has_locked_ridgeline: bool = False,
) -> List[List[float]]:
    """
    v10.5 全地貌三模態「100% 無重疊鑲嵌分區」多邊形生成器 (修復 P0-3 遮罩重疊覆蓋漏洞)：
    - 所有立面與中央背景下緣嚴格切齊至 road_y (0.72)，STREET_CORRIDOR 嚴格由 road_y 開始至 1.0，零重疊！
    1. DEMOLISH_TO_WATER_LEVEL (太古船塢、灣仔會展填海還原海域)：
       - 頂部天空與遠山鎖定 (0.0~0.36)，中層岸線分塊 (0.36~0.52)，下層填海轉水域 (0.52~1.0)。
    2. TRUNCATE_TO_LOW_RISE (銅鑼灣 5-6 層唐樓街景)：
       - 以實測消失點 vp_x 動態切分 xl / xr，五區精準鑲嵌填滿 1.0 x 1.0 畫布。
    3. KEEP_CANYON_MASS / EXTRUDE_HISTORICAL_MASS (時代廣場、曼克頓中央公園摩天樓)：
       - 左右立面直達頂部 (0.0~xl 與 xr~1.0)，中央 SKY_CANVAS 與 CENTRAL_BACKDROP 以互補幾何無重疊拼接。
    """
    xl = round(
        max(
            MIN_CORRIDOR_X_LEFT_BOUND,
            min(MAX_CORRIDOR_X_LEFT_BOUND, vp_x - CORRIDOR_HALF_SPAN_X),
        ),
        4,
    )
    xr = round(
        max(
            MIN_CORRIDOR_X_RIGHT_BOUND,
            min(MAX_CORRIDOR_X_RIGHT_BOUND, vp_x + CORRIDOR_HALF_SPAN_X),
        ),
        4,
    )
    outer_y = PERSPECTIVE_OUTER_ROOF_Y_RATIO
    road_y = DEFAULT_ROAD_TOP_Y_RATIO

    # 模態 A：填海地貌還原海域/船塢 (DEMOLISH_TO_WATER_LEVEL)
    if vertical_policy == VerticalMassPolicy.DEMOLISH_TO_WATER_LEVEL.value:
        sky_bottom_y = (
            WATERFRONT_SKY_Y_BOTTOM
            if has_locked_ridgeline
            else WATERFRONT_RIDGE_Y_BOTTOM
        )
        if topo_zone == TopologicalZoneType.SKY_CANVAS.value:
            return [
                [RATIO_MIN, RATIO_MIN],
                [RATIO_MAX, RATIO_MIN],
                [RATIO_MAX, sky_bottom_y],
                [RATIO_MIN, sky_bottom_y],
            ]
        if topo_zone == TopologicalZoneType.LOCKED_RIDGELINE.value:
            return [
                [RATIO_MIN, WATERFRONT_SKY_Y_BOTTOM],
                [RATIO_MAX, WATERFRONT_SKY_Y_BOTTOM],
                [RATIO_MAX, WATERFRONT_RIDGE_Y_BOTTOM],
                [RATIO_MIN, WATERFRONT_RIDGE_Y_BOTTOM],
            ]
        if topo_zone == TopologicalZoneType.LEFT_FACADE.value:
            return [
                [RATIO_MIN, WATERFRONT_RIDGE_Y_BOTTOM],
                [WATERFRONT_SHORE_X_SPAN, WATERFRONT_RIDGE_Y_BOTTOM],
                [WATERFRONT_SHORE_X_SPAN, WATERFRONT_WATER_Y_TOP],
                [RATIO_MIN, WATERFRONT_WATER_Y_TOP],
            ]
        if topo_zone == TopologicalZoneType.CENTRAL_BACKDROP.value:
            return [
                [WATERFRONT_SHORE_X_SPAN, WATERFRONT_RIDGE_Y_BOTTOM],
                [
                    RATIO_MAX - WATERFRONT_SHORE_X_SPAN,
                    WATERFRONT_RIDGE_Y_BOTTOM,
                ],
                [RATIO_MAX - WATERFRONT_SHORE_X_SPAN, WATERFRONT_WATER_Y_TOP],
                [WATERFRONT_SHORE_X_SPAN, WATERFRONT_WATER_Y_TOP],
            ]
        if topo_zone == TopologicalZoneType.RIGHT_FACADE.value:
            return [
                [
                    RATIO_MAX - WATERFRONT_SHORE_X_SPAN,
                    WATERFRONT_RIDGE_Y_BOTTOM,
                ],
                [RATIO_MAX, WATERFRONT_RIDGE_Y_BOTTOM],
                [RATIO_MAX, WATERFRONT_WATER_Y_TOP],
                [RATIO_MAX - WATERFRONT_SHORE_X_SPAN, WATERFRONT_WATER_Y_TOP],
            ]
        if topo_zone in (
            TopologicalZoneType.STREET_CORRIDOR.value,
            TopologicalZoneType.WATER_AND_DOCKYARD.value,
        ):
            return [
                [RATIO_MIN, WATERFRONT_WATER_Y_TOP],
                [RATIO_MAX, WATERFRONT_WATER_Y_TOP],
                [RATIO_MAX, RATIO_MAX],
                [RATIO_MIN, RATIO_MAX],
            ]

    # 模態 B：高樓降維至歷史低層唐樓街區 (TRUNCATE_TO_LOW_RISE) —— 100% 嚴格無重疊鑲嵌
    if enforce_sky_truncation:
        if topo_zone == TopologicalZoneType.SKY_CANVAS.value:
            return [
                [RATIO_MIN, RATIO_MIN],
                [RATIO_MAX, RATIO_MIN],
                [RATIO_MAX, outer_y],
                [xr, sky_cutoff_y],
                [xl, sky_cutoff_y],
                [RATIO_MIN, outer_y],
            ]
        if topo_zone == TopologicalZoneType.LEFT_FACADE.value:
            return [
                [RATIO_MIN, outer_y],
                [xl, sky_cutoff_y],
                [xl, road_y],
                [RATIO_MIN, road_y],
            ]
        if topo_zone == TopologicalZoneType.RIGHT_FACADE.value:
            return [
                [xr, sky_cutoff_y],
                [RATIO_MAX, outer_y],
                [RATIO_MAX, road_y],
                [xr, road_y],
            ]
        if topo_zone == TopologicalZoneType.CENTRAL_BACKDROP.value:
            return [
                [xl, sky_cutoff_y],
                [xr, sky_cutoff_y],
                [xr, road_y],
                [xl, road_y],
            ]
        if topo_zone == TopologicalZoneType.STREET_CORRIDOR.value:
            return [
                [RATIO_MIN, road_y],
                [RATIO_MAX, road_y],
                [RATIO_MAX, RATIO_MAX],
                [RATIO_MIN, RATIO_MAX],
            ]

    # 模態 C：歷史摩天大樓/都會峽谷/公園天際線模式 (KEEP_CANYON_MASS / EXTRUDE_HISTORICAL_MASS) —— 100% 嚴格無重疊鑲嵌
    mid_x = round(0.5 * (xl + xr), 4)
    c_l = round(
        max(xl + MIN_RATIO_SPAN, mid_x - CANYON_TOWER_HALF_SPAN_X), 4
    )
    c_r = round(
        min(xr - MIN_RATIO_SPAN, mid_x + CANYON_TOWER_HALF_SPAN_X), 4
    )

    if topo_zone == TopologicalZoneType.SKY_CANVAS.value:
        return [
            [xl, RATIO_MIN],
            [xr, RATIO_MIN],
            [xr, CANYON_SKY_SIDE_Y_BOTTOM],
            [c_r, CANYON_SKY_SIDE_Y_BOTTOM],
            [c_r, CANYON_CENTRAL_Y_TOP],
            [c_l, CANYON_CENTRAL_Y_TOP],
            [c_l, CANYON_SKY_SIDE_Y_BOTTOM],
            [xl, CANYON_SKY_SIDE_Y_BOTTOM],
        ]
    if topo_zone == TopologicalZoneType.LEFT_FACADE.value:
        return [
            [RATIO_MIN, RATIO_MIN],
            [xl, RATIO_MIN],
            [xl, road_y],
            [RATIO_MIN, road_y],
        ]
    if topo_zone == TopologicalZoneType.RIGHT_FACADE.value:
        return [
            [xr, RATIO_MIN],
            [RATIO_MAX, RATIO_MIN],
            [RATIO_MAX, road_y],
            [xr, road_y],
        ]
    if topo_zone == TopologicalZoneType.CENTRAL_BACKDROP.value:
        return [
            [xl, CANYON_SKY_SIDE_Y_BOTTOM],
            [c_l, CANYON_SKY_SIDE_Y_BOTTOM],
            [c_l, CANYON_CENTRAL_Y_TOP],
            [c_r, CANYON_CENTRAL_Y_TOP],
            [c_r, CANYON_SKY_SIDE_Y_BOTTOM],
            [xr, CANYON_SKY_SIDE_Y_BOTTOM],
            [xr, road_y],
            [xl, road_y],
        ]
    if topo_zone == TopologicalZoneType.STREET_CORRIDOR.value:
        return [
            [RATIO_MIN, road_y],
            [RATIO_MAX, road_y],
            [RATIO_MAX, RATIO_MAX],
            [RATIO_MIN, RATIO_MAX],
        ]

    x1, y1 = box_ratios["x_min_ratio"], box_ratios["y_min_ratio"]
    x2, y2 = box_ratios["x_max_ratio"], box_ratios["y_max_ratio"]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _sanitize_and_enrich_plan(
    plan_dict: Dict[str, Any],
    year: str,
    location: str,
    evidence_context: Dict[str, Any],
) -> Dict[str, Any]:
    safe_year = re.sub(r"[^a-zA-Z0-9]", "", str(year)) or "ERA"
    plan_id = f"PLAN_{safe_year}_{uuid.uuid4().hex[:8].upper()}"
    plan_dict["plan_id"] = plan_id

    evidence_card = evidence_context["evidence_card"]
    grounding_mode = evidence_card.get(
        "grounding_mode", "DIRECT_ARCHIVAL_EVIDENCE"
    )
    evidence_ref_id = evidence_card.get("evidence_ref_id", "EVID_DEFAULT_01")
    max_hist_stories = int(
        evidence_card.get("max_historical_stories", DEFAULT_LOW_RISE_MAX_STORIES)
    )
    input_is_daylight = bool(evidence_context.get("input_is_daylight", True))
    raw_palette = evidence_card.get("verified_color_palette_prompt", "")
    color_palette_prompt = _strip_haze_and_degrading_terms(
        raw_palette, enforce_daylight=input_is_daylight
    )
    is_proxy_mode = grounding_mode == "SPATIAL_TEMPORAL_PROXY"

    plan_dict["evidence_ref_id"] = evidence_ref_id
    plan_dict["grounding_mode"] = grounding_mode
    plan_dict["max_historical_stories"] = max_hist_stories
    plan_dict["verified_color_palette_prompt"] = color_palette_prompt
    plan_dict["archive_source_title"] = evidence_card.get(
        "archive_source_title", ""
    )
    plan_dict["sample_dir_indexed"] = evidence_context.get("sample_dir", "")
    plan_dict["reference_image_path"] = evidence_context.get(
        "reference_image_path"
    )
    plan_dict["reference_is_monochrome"] = evidence_context.get(
        "reference_is_monochrome", False
    )
    plan_dict["input_is_daylight"] = input_is_daylight
    plan_dict["input_luma"] = evidence_context.get(
        "input_luma", DEFAULT_NEUTRAL_LUMA
    )
    plan_dict["ipadapter_weight"] = float(
        evidence_card.get("ipadapter_weight", DEFAULT_IPADAPTER_WEIGHT)
    )
    plan_dict["ipadapter_end_at"] = float(
        evidence_card.get("ipadapter_end_at", DEFAULT_IPADAPTER_END_AT)
    )

    vis_arch = plan_dict.get("visual_archetype", {})
    if isinstance(vis_arch, dict):
        verified_arch = evidence_card.get("verified_architectural_features", "")
        vis_arch["architectural_prompt"] = _strip_haze_and_degrading_terms(
            f"{color_palette_prompt}, {verified_arch}",
            enforce_daylight=input_is_daylight,
        )

    req_policy = evidence_card.get("required_policy_matrix")
    if isinstance(req_policy, dict) and len(req_policy) == MIN_POLICY_MATRIX_KEYS:
        policy_matrix = dict(req_policy)
    else:
        policy_matrix = plan_dict.get("policy_matrix", {})
        if not isinstance(policy_matrix, dict):
            policy_matrix = {}

    backdrop_policy = policy_matrix.get("backdrop_sky_policy", "")
    vertical_policy = policy_matrix.get("vertical_mass_policy", "")
    plan_dict["policy_matrix"] = policy_matrix

    enforce_sky_truncation = (
        backdrop_policy == BackdropSkyPolicy.RELEASE_UPPER_SKY.value
        or vertical_policy == VerticalMassPolicy.TRUNCATE_TO_LOW_RISE.value
    )
    is_waterfront_lock_mode = (
        vertical_policy == VerticalMassPolicy.DEMOLISH_TO_WATER_LEVEL.value
        and backdrop_policy == BackdropSkyPolicy.LOCK_MOUNTAIN_RIDGELINE.value
    )

    geo = plan_dict.get("geography_profile", {})
    if not isinstance(geo, dict):
        geo = {}
    vp_x = max(
        VP_X_MIN_CLAMP,
        min(
            VP_X_MAX_CLAMP,
            _normalize_ratio(
                geo.get("vanishing_point_x_ratio", DEFAULT_VANISHING_POINT_X),
                DEFAULT_VANISHING_POINT_X,
            ),
        ),
    )
    vp_y = max(
        VP_Y_MIN_CLAMP,
        min(
            VP_Y_MAX_CLAMP,
            _normalize_ratio(
                geo.get("vanishing_point_y_ratio", DEFAULT_VANISHING_POINT_Y),
                DEFAULT_VANISHING_POINT_Y,
            ),
        ),
    )
    card_hw_ratio = float(
        evidence_card.get(
            "height_to_street_width_ratio", DEFAULT_HISTORICAL_CEILING_RATIO
        )
    )
    max_arch_h = (
        min(LOW_RISE_MAX_HEIGHT_CAP, card_hw_ratio)
        if enforce_sky_truncation
        else (
            card_hw_ratio
            if vertical_policy
            == VerticalMassPolicy.DEMOLISH_TO_WATER_LEVEL.value
            else max(CANYON_MIN_ARCH_HEIGHT_RATIO, card_hw_ratio)
        )
    )

    sky_cutoff_y_ratio = (
        round(max(MIN_SKY_CUTOFF_Y_RATIO, RATIO_MAX - max_arch_h), 4)
        if enforce_sky_truncation
        else (
            WATERFRONT_RIDGE_Y_BOTTOM
            if vertical_policy
            == VerticalMassPolicy.DEMOLISH_TO_WATER_LEVEL.value
            else CANYON_SKY_SIDE_Y_BOTTOM
        )
    )
    realistic_min_sky = (
        round(
            min(
                MAX_REQUIRED_SKY_VISIBILITY_CAP,
                max(
                    MIN_SKY_VISIBILITY_RATIO,
                    sky_cutoff_y_ratio * PERSPECTIVE_SKY_AREA_DISCOUNT,
                ),
            ),
            4,
        )
        if enforce_sky_truncation
        else CANYON_MIN_SKY_VISIBILITY_RATIO
    )
    geo["vanishing_point_x_ratio"] = vp_x
    geo["vanishing_point_y_ratio"] = vp_y
    geo["max_architecture_height_ratio"] = max_arch_h
    geo["outer_wing_max_height_ratio"] = (
        round(RATIO_MAX - PERSPECTIVE_OUTER_ROOF_Y_RATIO, 4)
        if enforce_sky_truncation
        else RATIO_MAX
    )
    geo["min_sky_visibility_ratio"] = realistic_min_sky
    plan_dict["geography_profile"] = geo

    landmarks = plan_dict.get("conjugate_landmarks", [])
    if not isinstance(landmarks, list) or len(landmarks) == 0:
        landmarks = [
            {
                "landmark_id": "anchor_corridor_vp",
                "landmark_description": "Primary street corridor perspective vanishing point",
                "x_ratio": vp_x,
                "y_ratio": vp_y,
                "cross_era_stability_confidence": DEFAULT_LOCKED_ZONE_CONFIDENCE,
            }
        ]
    plan_dict["conjugate_landmarks"] = landmarks
    primary_anchor_id = landmarks[0].get("landmark_id", "anchor_corridor_vp")

    contracts = plan_dict.get("region_contracts", [])
    sanitized_contracts: List[Dict[str, Any]] = []
    seen_zones: Set[str] = set()

    for idx, contract in enumerate(contracts):
        if not isinstance(contract, dict):
            continue

        sem_cls = str(contract.get("semantic_class", "structure"))
        topo_zone = _normalize_topological_zone(
            contract.get("topological_zone"), sem_cls
        )
        # 在非水岸模式或已有 STREET_CORRIDOR 時避免重複 WATER_AND_DOCKYARD 疊加
        if (
            topo_zone == TopologicalZoneType.WATER_AND_DOCKYARD.value
            and TopologicalZoneType.STREET_CORRIDOR.value in seen_zones
        ):
            continue
        if (
            topo_zone == TopologicalZoneType.STREET_CORRIDOR.value
            and TopologicalZoneType.WATER_AND_DOCKYARD.value in seen_zones
        ):
            continue
        if topo_zone in seen_zones:
            continue
        seen_zones.add(topo_zone)
        contract["topological_zone"] = topo_zone

        raw_box = _sanitize_box_ratios(
            contract.get("x_min_ratio", RATIO_MIN),
            contract.get("y_min_ratio", RATIO_MIN),
            contract.get("x_max_ratio", RATIO_MAX),
            contract.get("y_max_ratio", RATIO_MAX),
        )
        aligned_poly = _build_perspective_aligned_polygon(
            topo_zone=topo_zone,
            vertical_policy=vertical_policy,
            enforce_sky_truncation=enforce_sky_truncation,
            sky_cutoff_y=sky_cutoff_y_ratio,
            vp_x=vp_x,
            vp_y=vp_y,
            box_ratios=raw_box,
            has_locked_ridgeline=is_waterfront_lock_mode,
        )
        _apply_polygon_and_sync_fields(contract, aligned_poly)

        layer_idx = int(contract.get("layer_index", LAYER_INDEX_MESO_STRUCTURE))
        contract["layer_index"] = max(
            LAYER_INDEX_MACRO_BACKDROP, min(LAYER_INDEX_MICRO_ENTITIES, layer_idx)
        )

        if topo_zone in (
            TopologicalZoneType.STREET_CORRIDOR.value,
            TopologicalZoneType.WATER_AND_DOCKYARD.value,
        ):
            zone_pos = evidence_card.get("verified_street_features", "")
        elif topo_zone in (
            TopologicalZoneType.SKY_CANVAS.value,
            TopologicalZoneType.LOCKED_RIDGELINE.value,
        ):
            zone_pos = "clear natural daylight blue sky, realistic atmospheric horizon"
        else:
            zone_pos = evidence_card.get("verified_architectural_features", "")

        contract["regional_positive_prompt"] = _strip_haze_and_degrading_terms(
            zone_pos, enforce_daylight=input_is_daylight
        )

        (
            change_op,
            substrate,
            act_type,
            mask_mode,
            ctrl_rep,
            post_rule,
        ) = _derive_operator_and_substrate(
            topo_zone=topo_zone,
            raw_operator=contract.get("change_operator"),
            vertical_policy=vertical_policy,
            backdrop_policy=backdrop_policy,
            evidence_card=evidence_card,
        )

        contract["change_operator"] = change_op
        contract["target_substrate"] = substrate
        contract["anchor_landmark_id"] = (
            contract.get("anchor_landmark_id") or primary_anchor_id
        )
        contract["relative_topology_relation"] = (
            contract.get("relative_topology_relation")
            or f"{change_op}_RELATIVE_TO_{primary_anchor_id.upper()}"
        )
        contract["evidence_ref_id"] = evidence_ref_id
        contract["uncertainty_flag"] = bool(
            contract.get("uncertainty_flag", False) or is_proxy_mode
        )
        contract["uncertainty_reason"] = (
            "SPATIAL_TEMPORAL_NEIGHBOR_PROXY"
            if contract["uncertainty_flag"]
            else "VERIFIED_BY_ARCHIVE_CARD"
        )
        contract["action_type"] = act_type
        contract["inpaint_mask_mode"] = mask_mode
        contract["control_representation"] = ctrl_rep
        contract["post_composite_rule"] = post_rule
        contract["generator_inputs"] = {
            "inpaint_mask_mode": mask_mode,
            "control_representations": [ctrl_rep],
            "post_composite_rule": post_rule,
        }

        conf = float(contract.get("confidence", DEFAULT_CONTRACT_CONFIDENCE))
        if is_proxy_mode:
            conf = min(PROXY_MODE_CONFIDENCE_CAP, conf)
        contract["confidence"] = max(RATIO_MIN, min(RATIO_MAX, conf))
        contract["requires_human_confirmation"] = (
            contract["uncertainty_flag"]
            or contract["confidence"] < HUMAN_CONFIRM_CONFIDENCE_THRESHOLD
        )

        if not contract.get("region_id"):
            contract["region_id"] = f"reg_{topo_zone.lower()}_{idx}"
        sanitized_contracts.append(contract)

    required_zones: List[str] = [
        TopologicalZoneType.SKY_CANVAS.value,
        TopologicalZoneType.LEFT_FACADE.value,
        TopologicalZoneType.CENTRAL_BACKDROP.value,
        TopologicalZoneType.RIGHT_FACADE.value,
    ]
    if is_waterfront_lock_mode:
        required_zones.append(TopologicalZoneType.LOCKED_RIDGELINE.value)
    if TopologicalZoneType.WATER_AND_DOCKYARD.value not in seen_zones:
        required_zones.append(TopologicalZoneType.STREET_CORRIDOR.value)

    for req_z in required_zones:
        if req_z not in seen_zones:
            (
                c_op,
                c_sub,
                c_act,
                c_mask,
                c_ctrl,
                c_post,
            ) = _derive_operator_and_substrate(
                topo_zone=req_z,
                raw_operator=None,
                vertical_policy=vertical_policy,
                backdrop_policy=backdrop_policy,
                evidence_card=evidence_card,
            )
            auto_contract: Dict[str, Any] = {
                "region_id": f"reg_{req_z.lower()}_auto",
                "layer_index": (
                    LAYER_INDEX_MACRO_BACKDROP
                    if req_z
                    in (
                        TopologicalZoneType.SKY_CANVAS.value,
                        TopologicalZoneType.LOCKED_RIDGELINE.value,
                    )
                    else LAYER_INDEX_MESO_STRUCTURE
                ),
                "topological_zone": req_z,
                "change_operator": c_op,
                "target_substrate": c_sub,
                "anchor_landmark_id": primary_anchor_id,
                "relative_topology_relation": f"{c_op}_{req_z}",
                "evidence_ref_id": evidence_ref_id,
                "uncertainty_flag": is_proxy_mode,
                "uncertainty_reason": (
                    "SPATIAL_TEMPORAL_NEIGHBOR_PROXY"
                    if is_proxy_mode
                    else "VERIFIED_BY_ARCHIVE_CARD"
                ),
                "semantic_class": req_z.lower(),
                "action_type": c_act,
                "historical_evidence_source": evidence_card.get(
                    "archive_source_title", ""
                ),
                "regional_positive_prompt": _strip_haze_and_degrading_terms(
                    evidence_card.get("verified_street_features", "")
                    if req_z == TopologicalZoneType.STREET_CORRIDOR.value
                    else evidence_card.get("verified_architectural_features", ""),
                    enforce_daylight=input_is_daylight,
                ),
                "regional_negative_prompt": "modern skyscrapers, LED screens, modern clothing, smartphones, monochrome, black and white, night scene",
                "confidence": DEFAULT_CONTRACT_CONFIDENCE,
                "requires_human_confirmation": is_proxy_mode,
                "inpaint_mask_mode": c_mask,
                "control_representation": c_ctrl,
                "post_composite_rule": c_post,
                "generator_inputs": {
                    "inpaint_mask_mode": c_mask,
                    "control_representations": [c_ctrl],
                    "post_composite_rule": c_post,
                },
            }
            auto_poly = _build_perspective_aligned_polygon(
                topo_zone=req_z,
                vertical_policy=vertical_policy,
                enforce_sky_truncation=enforce_sky_truncation,
                sky_cutoff_y=sky_cutoff_y_ratio,
                vp_x=vp_x,
                vp_y=vp_y,
                box_ratios=_sanitize_box_ratios(
                    RATIO_MIN, RATIO_MIN, RATIO_MAX, RATIO_MAX
                ),
                has_locked_ridgeline=is_waterfront_lock_mode,
            )
            _apply_polygon_and_sync_fields(auto_contract, auto_poly)
            sanitized_contracts.append(auto_contract)

    sanitized_contracts.sort(key=lambda c: c["layer_index"])
    plan_dict["region_contracts"] = sanitized_contracts

    protected_zones = []
    for c in sanitized_contracts:
        if c.get("change_operator") == SceneChangeOperator.KEEP.value:
            protected_zones.append(
                {
                    "category": c.get("semantic_class", "LOCKED_LANDMARK"),
                    "confidence": c.get(
                        "confidence", DEFAULT_LOCKED_ZONE_CONFIDENCE
                    ),
                    "bbox": c["bbox"],
                }
            )
    plan_dict["protected_zones"] = protected_zones
    return plan_dict


async def _call_gemini_with_fallback(
    contents: List[Any],
    system_instruction: str,
    response_schema: Any,
    temperature: float,
    task_label: str,
    prefer_strong_auditor: bool = False,
) -> Optional[Dict[str, Any]]:
    if not client:
        logger.error(f"🚨 [{task_label}] Gemini Client 未初始化。")
        return None

    raw_queue = (
        list(CANDIDATE_MODELS_QUEUE)
        if CANDIDATE_MODELS_QUEUE
        else await get_candidate_models(client)
    )
    candidate_models = [
        m for m in raw_queue if m and m not in DEAD_MODELS_BLACKLIST
    ]
    if not candidate_models:
        candidate_models = list(FALLBACK_GEMINI_MODELS)

    if prefer_strong_auditor:
        candidate_models.sort(key=lambda m: (1 if "lite" in m else 0))
    elif ACTIVE_MODEL and ACTIVE_MODEL not in DEAD_MODELS_BLACKLIST:
        if ACTIVE_MODEL in candidate_models:
            candidate_models.remove(ACTIVE_MODEL)
        candidate_models.insert(0, ACTIVE_MODEL)

    last_error = None
    for model_name in candidate_models:
        try:
            logger.info(f"🧠 [{task_label}] 執行非同步模型: {model_name}")
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=temperature,
                ),
            )
            raw_text = _clean_json_markdown(response.text)
            return json.loads(raw_text)
        except Exception as e:
            err_str = str(e)
            last_error = e
            if "404" in err_str or "NOT_FOUND" in err_str:
                DEAD_MODELS_BLACKLIST.add(model_name)
                if model_name in CANDIDATE_MODELS_QUEUE:
                    CANDIDATE_MODELS_QUEUE.remove(model_name)
                logger.warning(
                    f"🚫 [{task_label}] 模型 {model_name} 回傳 404，已永久移出候選佇列。"
                )
            else:
                logger.warning(
                    f"⚠️ [{task_label}] 模型 {model_name} 執行異常，切換下一候選模型: {err_str}"
                )

    logger.error(
        f"❌ [{task_label}] 所有候選模型均失敗，最後錯誤: {str(last_error)}"
    )
    return None


# ==============================================================================
# 7. 核心接口一：Stage 1 歷史證據驅動之跨年代場景變化圖規劃器
# ==============================================================================
async def plan_historical_reconstruction(
    image_bytes: bytes, year: str, location: str
) -> Dict[str, Any]:
    evidence_context = retrieve_historical_evidence_context(
        location=location, year=year, input_image_bytes=image_bytes
    )
    evidence_card = evidence_context["evidence_card"]
    ref_img_bytes = evidence_context.get("reference_image_bytes")
    ref_is_mono = evidence_context.get("reference_is_monochrome", False)
    input_is_daylight = evidence_context.get("input_is_daylight", True)

    mime_type = _detect_mime_type(image_bytes)
    contents_list: List[Any] = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    ]
    if ref_img_bytes is not None:
        ref_mime = _detect_mime_type(ref_img_bytes)
        contents_list.append(
            types.Part.from_bytes(data=ref_img_bytes, mime_type=ref_mime)
        )

    mono_notice = (
        "CRITICAL COLOR DECOUPLING RULE: The attached historical reference photo is monochrome/black-and-white. "
        "Use it ONLY for structural massing and layout! You MUST specify FULL-COLOR Kodachrome descriptions!"
        if ref_is_mono
        else "Use the structural proportions and authentic vintage color palette from the attached historical reference photo."
    )
    daylight_notice = (
        "CRITICAL DAYLIGHT INVARIANCE RULE: The user's input photo is taken in BRIGHT DAYLIGHT. "
        "Even if the historical reference photo was taken at dusk or night, you MUST specify BRIGHT DAYTIME SUNLIGHT and NEVER output dusk, night, dark facades, or vehicle headlights!"
        if input_is_daylight
        else "Preserve the ambient lighting time-of-day of the user's input photo."
    )

    system_instruction = f"""You are the Chief Historical Geomorphic Reconstruction Planner for a commercial Time Machine App.
Target Era: {year}. Target Location: {location}.
{mono_notice}
{daylight_notice}

VERIFIED HISTORICAL EVIDENCE CARD TO ENFORCE:
- Evidence Ref ID: {evidence_card['evidence_ref_id']} ({evidence_card['grounding_mode']})
- Required 4D Policy Matrix: {json.dumps(evidence_card.get('required_policy_matrix', {}))}
- Max Historical Building Stories Allowed: {evidence_card['max_historical_stories']} stories
- Verified Color Palette: {evidence_card.get('verified_color_palette_prompt', '')}
- Required 4-Operator Mapping (`change_operator`): {json.dumps(evidence_card['operator_mapping_rules'])}
- Verified Architectural Typology: {evidence_card['verified_architectural_features']}
- Verified Street, Vehicles & Pedestrian Attire Features: {evidence_card['verified_street_features']}"""

    prompt = (
        f"Formulate the v10.5 Cross-Era Scene Change Graph for {location} in {year} "
        f"strictly grounded in Evidence ID {evidence_card['evidence_ref_id']}."
    )
    contents_list.append(prompt)

    plan_dict = await _call_gemini_with_fallback(
        contents=contents_list,
        system_instruction=system_instruction,
        response_schema=ReconstructionPlanResponse,
        temperature=VLM_PLANNER_TEMPERATURE,
        task_label="Stage1-Planner",
        prefer_strong_auditor=False,
    )

    if not plan_dict:
        return {
            "pass": False,
            "reason": "VLM Planner 服務無法取得有效回應，啟動安全跳過",
            "bboxes": [],
            "error_level": "SYSTEM_WARNING_SKIP",
            "raw_semantic_data": {},
        }

    plan_dict = _sanitize_and_enrich_plan(
        plan_dict, year, location, evidence_context
    )
    is_already_authentic = bool(
        plan_dict.get("already_historically_authentic", False)
    )
    error_level = "NONE" if is_already_authentic else "CONTENT_REJECT"
    plan_dict["pass_validation"] = is_already_authentic
    plan_dict["error_level"] = error_level

    legacy_bboxes = [z["bbox"] for z in plan_dict.get("protected_zones", [])]
    reasoning_str = " | ".join(plan_dict.get("reasoning_steps", []))

    op_summary = {
        c["topological_zone"]: c["change_operator"]
        for c in plan_dict.get("region_contracts", [])
    }
    logger.info(
        f"🗺️ [Stage 1 跨年代場景變化圖完成] Plan ID: {plan_dict['plan_id']} | "
        f"Evidence ID: {plan_dict.get('evidence_ref_id')} (上限={plan_dict.get('max_historical_stories')}層) | "
        f"4D Policy: {plan_dict.get('policy_matrix')} | 四算子配置: {op_summary}"
    )

    return {
        "pass": is_already_authentic,
        "reason": reasoning_str,
        "bboxes": legacy_bboxes,
        "error_level": error_level,
        "raw_semantic_data": plan_dict,
        "reference_image_path": evidence_context.get("reference_image_path"),
        "reference_image_bytes": ref_img_bytes,
        "reference_is_monochrome": ref_is_mono,
        "color_reference_bytes": evidence_context.get("color_reference_bytes"),
        "input_is_daylight": input_is_daylight,
    }


# ==============================================================================
# 8. 核心接口二：Stage 4 三圖對照盲測取證 + 無連坐分區核驗 + 歷史參考實證記錄
# ==============================================================================
def _sanitize_evidence_bbox_1000(raw_xyxy: Any) -> List[int]:
    if not isinstance(raw_xyxy, list) or len(raw_xyxy) < 4:
        return [0, 0, LEGACY_COORD_SCALE, LEGACY_COORD_SCALE]
    box = _sanitize_box_ratios(raw_xyxy[0], raw_xyxy[1], raw_xyxy[2], raw_xyxy[3])
    return [
        int(round(box["x_min_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box["y_min_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box["x_max_ratio"] * LEGACY_COORD_SCALE)),
        int(round(box["y_max_ratio"] * LEGACY_COORD_SCALE)),
    ]


async def verify_reconstruction_dual_track(
    original_image_bytes: bytes,
    generated_image_bytes: bytes,
    year: str,
    location: str,
    plan_data: Optional[Dict[str, Any]] = None,
    cv_compliance: Optional[Dict[str, Any]] = None,
    clarity_score: Optional[float] = None,
    reference_image_bytes: Optional[bytes] = None,
    historical_reference_applied: bool = False,
) -> Dict[str, Any]:
    orig_mime = _detect_mime_type(original_image_bytes)
    gen_mime = _detect_mime_type(generated_image_bytes)

    orig_part = types.Part.from_bytes(
        data=original_image_bytes, mime_type=orig_mime
    )
    gen_part = types.Part.from_bytes(
        data=generated_image_bytes, mime_type=gen_mime
    )

    contents_list: List[Any] = [orig_part]
    has_ref_image = reference_image_bytes is not None
    if has_ref_image:
        ref_mime = _detect_mime_type(reference_image_bytes)
        ref_part = types.Part.from_bytes(
            data=reference_image_bytes, mime_type=ref_mime
        )
        contents_list.append(ref_part)
        contents_list.append(gen_part)
        image_role_desc = (
            "IMAGE 1 = Modern Input Photo (for road width and perspective alignment).\n"
            "IMAGE 2 = Authentic Historical Reference Photo from Archive (ground truth for era architecture, stories, signage, vehicles, and pedestrian attire).\n"
            "IMAGE 3 = Candidate Generated Image to audit."
        )
        candidate_label = "Image 3"
    else:
        contents_list.append(gen_part)
        image_role_desc = (
            "IMAGE 1 = Modern Input Photo (for road width and perspective alignment).\n"
            "IMAGE 2 = Candidate Generated Image to audit."
        )
        candidate_label = "Image 2"

    policy_matrix = (plan_data or {}).get("policy_matrix", {})
    geo_profile = (plan_data or {}).get("geography_profile", {})
    plan_id = (plan_data or {}).get(
        "plan_id", f"AUDIT_{uuid.uuid4().hex[:6].upper()}"
    )
    evidence_ref_id = (plan_data or {}).get("evidence_ref_id", "EVID_DEFAULT_01")
    allowed_max_stories = int(
        (plan_data or {}).get(
            "max_historical_stories", DEFAULT_LOW_RISE_MAX_STORIES
        )
    )
    input_is_daylight = bool((plan_data or {}).get("input_is_daylight", True))

    target_max_height = float(
        geo_profile.get(
            "max_architecture_height_ratio", DEFAULT_HISTORICAL_CEILING_RATIO
        )
    )
    outer_wing_max_height = float(
        geo_profile.get(
            "outer_wing_max_height_ratio",
            round(RATIO_MAX - PERSPECTIVE_OUTER_ROOF_Y_RATIO, 4),
        )
    )
    target_min_sky = float(
        geo_profile.get("min_sky_visibility_ratio", MIN_SKY_VISIBILITY_RATIO)
    )

    system_instruction = f"""You are a Strict Tri-Modal Forensic Visual Auditor for a Historical Time Machine App.
Target Historical Era & Location to verify: {location} in {year} (Evidence Ref: {evidence_ref_id}).
{image_role_desc}

CRITICAL BLIND FORENSIC RULES:
1. FULL-COLOR, DAYLIGHT & PHOTOREALISM CHECK (`visual_evidence_inventory`):
   - Inspect {candidate_label} carefully. Set `is_full_color_not_monochrome = false` if {candidate_label} is black-and-white, greyscale, or monochromatic!
   - Set `daylight_illumination_preserved = false` if IMAGE 1 is daytime but {candidate_label} was turned into a dark night/dusk scene!
   - If {candidate_label} looks like a cartoon, vector illustration, vintage poster, or has artificial flat digital color blocks, set `is_photorealistic_documentary_photo = false`!
2. EXACT BBOX, STORY COUNT, VEHICLES & PEDESTRIANS CHECK (`visual_evidence_inventory`):
   - Measure `tallest_building_bbox_xyxy` `[x_min, y_min, x_max, y_max]` (0.0 to 1.0) and `estimated_max_building_stories` of the tallest building in {candidate_label}.
   - Set `vehicles_structurally_realistic_and_on_lanes = false` if modern 2000s/2020s vehicles (e.g., Ford Crown Victoria taxi, modern LED bus/tram) remain unrewritten or if vehicles are deformed.
   - Set `pedestrians_attire_and_appearance_authentic = false` if pedestrians wear modern 21st-century clothing/backpacks or look like distorted/abstract ghost figures!
3. ZONE-BY-ZONE INDEPENDENT VERDICT (`zone_audits`):
   - Evaluate all 5 zones (`SKY_CANVAS`, `LEFT_FACADE`, `CENTRAL_BACKDROP`, `RIGHT_FACADE`, `STREET_CORRIDOR`) in {candidate_label} independently against the Historical Reference and Evidence Card.
   - Flag any zone as `zone_passed = false` if it merely copied modern elements from IMAGE 1 without genuine historical transformation."""

    prompt = (
        f"Perform a strict forensic visual audit on {candidate_label} (Candidate for {location} in {year}) "
        f"using {'Tri-Image comparison (Modern + Historical Reference + Candidate)' if has_ref_image else 'Dual-Image comparison'}. "
        f"Output strict DualTrackQAResponse JSON."
    )
    contents_list.append(prompt)

    qa_dict = await _call_gemini_with_fallback(
        contents=contents_list,
        system_instruction=system_instruction,
        response_schema=DualTrackQAResponse,
        temperature=VLM_AUDITOR_TEMPERATURE,
        task_label="Stage4-TriModalQA",
        prefer_strong_auditor=True,
    )

    if not qa_dict:
        return {
            "pass": False,
            "qa_compliance_score": RATIO_MIN,
            "passing_zone_count": 0,
            "pass_scene_identity": False,
            "is_photorealistic": False,
            "daylight_ok": False,
            "vehicles_realistic": False,
            "pedestrians_authentic": False,
            "historical_reference_applied": historical_reference_applied,
            "measured_building_height_ratio": DEFAULT_FALLBACK_HEIGHT_RATIO,
            "estimated_max_building_stories": 0,
            "has_valid_global_foundation": False,
            "zone_verification_map": {},
            "reason": "Tri-Modal QA 服務無法取得有效回應",
            "bboxes": [],
            "error_level": "SYSTEM_WARNING_SKIP",
            "primary_defect_zone": "NONE",
            "failed_region_ids": [],
            "failed_topological_zones": [],
            "raw_semantic_data": plan_data or {},
        }

    inv = qa_dict.get("visual_evidence_inventory", {})
    s_id = qa_dict.get("scene_identity_audit", {})
    t1 = qa_dict.get("track1_audit", {})
    t2 = qa_dict.get("track2_audit", {})
    zone_audits = qa_dict.get("zone_audits", [])

    media_style = str(
        inv.get(
            "media_style_classification",
            MediaStyleClassification.AUTHENTIC_PHOTOGRAPH.value,
        )
    ).upper()
    is_full_color = bool(inv.get("is_full_color_not_monochrome", True))
    daylight_ok = (
        bool(inv.get("daylight_illumination_preserved", True))
        if input_is_daylight
        else True
    )
    is_photorealistic = (
        bool(inv.get("is_photorealistic_documentary_photo", False))
        and is_full_color
        and media_style == MediaStyleClassification.AUTHENTIC_PHOTOGRAPH.value
    )
    vehicles_realistic = bool(
        inv.get("vehicles_structurally_realistic_and_on_lanes", False)
    )
    pedestrians_authentic = bool(
        inv.get("pedestrians_attire_and_appearance_authentic", True)
    )
    estimated_stories = int(
        inv.get("estimated_max_building_stories", DEFAULT_ESTIMATED_STORIES)
    )

    tallest_b_xyxy = inv.get(
        "tallest_building_bbox_xyxy", list(DEFAULT_TALLEST_BUILDING_BBOX)
    )
    tallest_b_box = _sanitize_box_ratios(
        tallest_b_xyxy[0]
        if len(tallest_b_xyxy) > 0
        else DEFAULT_TALLEST_BUILDING_BBOX[0],
        tallest_b_xyxy[1]
        if len(tallest_b_xyxy) > 1
        else DEFAULT_TALLEST_BUILDING_BBOX[1],
        tallest_b_xyxy[2]
        if len(tallest_b_xyxy) > 2
        else DEFAULT_TALLEST_BUILDING_BBOX[2],
        tallest_b_xyxy[3]
        if len(tallest_b_xyxy) > 3
        else DEFAULT_TALLEST_BUILDING_BBOX[3],
    )
    bbox_derived_height = round(RATIO_MAX - tallest_b_box["y_min_ratio"], 4)
    reported_height = _normalize_ratio(
        t1.get("measured_building_height_ratio", bbox_derived_height),
        bbox_derived_height,
    )
    measured_height = max(bbox_derived_height, reported_height)
    t1["measured_building_height_ratio"] = measured_height

    bbox_center_x = 0.5 * (
        tallest_b_box["x_min_ratio"] + tallest_b_box["x_max_ratio"]
    )
    is_outer_perspective_wing = (
        bbox_center_x < PERSPECTIVE_INNER_CORRIDOR_X_LEFT
        or bbox_center_x > PERSPECTIVE_INNER_CORRIDOR_X_RIGHT
    )
    effective_allowed_height = (
        outer_wing_max_height
        if is_outer_perspective_wing
        else target_max_height
    )
    if bbox_center_x < PERSPECTIVE_INNER_CORRIDOR_X_LEFT:
        tallest_building_zone = TopologicalZoneType.LEFT_FACADE.value
    elif bbox_center_x > PERSPECTIVE_INNER_CORRIDOR_X_RIGHT:
        tallest_building_zone = TopologicalZoneType.RIGHT_FACADE.value
    else:
        tallest_building_zone = TopologicalZoneType.CENTRAL_BACKDROP.value

    measured_sky = _normalize_ratio(
        t1.get("measured_sky_visibility_ratio", RATIO_MIN), RATIO_MIN
    )

    zone_verification_map: Dict[str, bool] = {
        TopologicalZoneType.SKY_CANVAS.value: True,
        TopologicalZoneType.LEFT_FACADE.value: True,
        TopologicalZoneType.CENTRAL_BACKDROP.value: True,
        TopologicalZoneType.RIGHT_FACADE.value: True,
        TopologicalZoneType.STREET_CORRIDOR.value: True,
    }
    failed_zones: List[str] = list(qa_dict.get("failed_topological_zones", []))

    for z_item in zone_audits:
        if isinstance(z_item, dict):
            z_name = _normalize_topological_zone(
                z_item.get("topological_zone"), ""
            )
            z_pass = bool(z_item.get("zone_passed", False))
            zone_verification_map[z_name] = z_pass
            if not z_pass and z_name not in failed_zones:
                failed_zones.append(z_name)

    backdrop_policy = policy_matrix.get("backdrop_sky_policy", "")
    vertical_policy = policy_matrix.get("vertical_mass_policy", "")
    enforce_low_rise_truncation = (
        backdrop_policy == BackdropSkyPolicy.RELEASE_UPPER_SKY.value
        or vertical_policy == VerticalMassPolicy.TRUNCATE_TO_LOW_RISE.value
    )

    override_reasons: List[str] = []

    if not is_full_color:
        t1["render_integrity_and_color_valid"] = False
        override_reasons.append(
            "[色彩空間一票否決] 畫面退化為黑白或灰階單色調，違反全彩懷舊照片要求"
        )

    if not daylight_ok:
        t1["render_integrity_and_color_valid"] = False
        override_reasons.append(
            "[光照守恆攔截] 原圖為日間場景，但生成圖被錯誤拉暗為黃昏或夜景"
        )

    if not is_photorealistic:
        t1["render_integrity_and_color_valid"] = False
        override_reasons.append(
            f"[寫實度一票否決] 畫面呈現非寫實照片風格或含人工色塊遮罩 (檢測分類={media_style}, 全彩={is_full_color})"
        )
        for z_k in (
            TopologicalZoneType.LEFT_FACADE.value,
            TopologicalZoneType.RIGHT_FACADE.value,
            TopologicalZoneType.STREET_CORRIDOR.value,
        ):
            zone_verification_map[z_k] = False
            if z_k not in failed_zones:
                failed_zones.append(z_k)

    if (
        clarity_score is not None
        and float(clarity_score) < MIN_PASSING_CLARITY_SCORE
    ):
        t1["render_integrity_and_color_valid"] = False
        zone_verification_map[TopologicalZoneType.STREET_CORRIDOR.value] = False
        if TopologicalZoneType.STREET_CORRIDOR.value not in failed_zones:
            failed_zones.insert(0, TopologicalZoneType.STREET_CORRIDOR.value)
        override_reasons.append(
            f"[清晰度硬門檻攔截] 實測非天空區清晰度 ({clarity_score:.1f}) 低於最低寫實門檻 ({MIN_PASSING_CLARITY_SCORE})"
        )

    if not vehicles_realistic:
        t2["signage_and_vehicles_valid"] = False
        zone_verification_map[TopologicalZoneType.STREET_CORRIDOR.value] = False
        if TopologicalZoneType.STREET_CORRIDOR.value not in failed_zones:
            failed_zones.insert(0, TopologicalZoneType.STREET_CORRIDOR.value)
        override_reasons.append(
            "[車輛結構與年代攔截] 車輛或電車仍為現代車款、無軌懸浮或比例畸變"
        )

    if not pedestrians_authentic:
        t2["signage_and_vehicles_valid"] = False
        zone_verification_map[TopologicalZoneType.STREET_CORRIDOR.value] = False
        if TopologicalZoneType.STREET_CORRIDOR.value not in failed_zones:
            failed_zones.insert(0, TopologicalZoneType.STREET_CORRIDOR.value)
        override_reasons.append(
            "[行人衣著與形態攔截] 行人仍穿著現代服飾或人物形態模糊抽象"
        )

    # 防輪迴黑名單 #10：解除跨分區連坐否決 —— 僅標記實際超高建築所屬分區，絕不連坐抹殺已合格之 SKY_CANVAS！
    if enforce_low_rise_truncation:
        if estimated_stories > allowed_max_stories or measured_height > (
            effective_allowed_height + HEIGHT_ENVELOPE_TOLERANCE
        ):
            t1["envelope_and_sky_compliant"] = False
            t2["architectural_typology_verified"] = False
            zone_verification_map[tallest_building_zone] = False
            if tallest_building_zone not in failed_zones:
                failed_zones.insert(0, tallest_building_zone)
            override_reasons.append(
                f"[樓層與透視包絡攔截] {tallest_building_zone} 出現約 {estimated_stories} 層高樓 "
                f"(高度佔比={measured_height:.2f})，超出低層證據卡上限 ({allowed_max_stories} 層 / 允許佔比<={effective_allowed_height:.2f})"
            )

        if measured_sky < target_min_sky:
            t1["envelope_and_sky_compliant"] = False
            zone_verification_map[TopologicalZoneType.SKY_CANVAS.value] = False
            if TopologicalZoneType.SKY_CANVAS.value not in failed_zones:
                failed_zones.insert(0, TopologicalZoneType.SKY_CANVAS.value)
            override_reasons.append(
                f"[P2 天際線不變量攔截] 實測天空可見度 ({measured_sky:.2f}) 低於要求 ({target_min_sky:.2f})"
            )
    else:
        t1["envelope_and_sky_compliant"] = True
        zone_verification_map[TopologicalZoneType.SKY_CANVAS.value] = True
        if TopologicalZoneType.SKY_CANVAS.value in failed_zones:
            failed_zones.remove(TopologicalZoneType.SKY_CANVAS.value)

    modern_artifacts = t1.get("modern_artifacts_detected", [])
    if modern_artifacts:
        t1["render_integrity_and_color_valid"] = False
        zone_verification_map[TopologicalZoneType.LEFT_FACADE.value] = False
        if TopologicalZoneType.LEFT_FACADE.value not in failed_zones:
            failed_zones.append(TopologicalZoneType.LEFT_FACADE.value)
        override_reasons.append(
            f"[現代殘留攔截] 發現未清除現代元素: {', '.join(modern_artifacts)}"
        )

    haze_washout = False
    if isinstance(cv_compliance, dict):
        if enforce_low_rise_truncation and not bool(
            cv_compliance.get("cv_sky_compliant", True)
        ):
            t1["envelope_and_sky_compliant"] = False
            zone_verification_map[TopologicalZoneType.SKY_CANVAS.value] = False
            if TopologicalZoneType.SKY_CANVAS.value not in failed_zones:
                failed_zones.insert(0, TopologicalZoneType.SKY_CANVAS.value)
            override_reasons.append(
                "[CV 物理攔截] 頂部天際區殘留過多建築結構邊緣，違反 RELEASE_UPPER_SKY"
            )
        if bool(cv_compliance.get("haze_washout_detected", False)):
            haze_washout = True
            t1["render_integrity_and_color_valid"] = False
            zone_verification_map[TopologicalZoneType.STREET_CORRIDOR.value] = (
                False
            )
            if TopologicalZoneType.STREET_CORRIDOR.value not in failed_zones:
                failed_zones.insert(
                    0, TopologicalZoneType.STREET_CORRIDOR.value
                )
            override_reasons.append(
                "[CV 物理攔截] 畫面對比度或色彩飽和度過低，存在明顯霧化或黑白退化"
            )
        if bool(cv_compliance.get("unrewritten_central_void_detected", False)):
            t1["render_integrity_and_color_valid"] = False
            for u_zone in cv_compliance.get(
                "under_restyled_zones",
                [TopologicalZoneType.CENTRAL_BACKDROP.value],
            ):
                if u_zone == TopologicalZoneType.SKY_CANVAS.value:
                    continue
                zone_verification_map[u_zone] = False
                if u_zone not in failed_zones:
                    failed_zones.insert(0, u_zone)
            override_reasons.append(
                f"[CV 原圖複製攔截] 非天空分區像素變化過低 ({cv_compliance.get('under_restyled_zones', [])})，存在未重繪原圖殘留"
            )

    pass_identity = (
        bool(s_id.get("pass_scene_identity", False))
        and bool(s_id.get("road_scale_and_width_preserved", False))
        and bool(s_id.get("regional_urban_identity_verified", False))
        and bool(s_id.get("no_village_or_alley_degradation", False))
    )
    if not pass_identity:
        id_reason = s_id.get("identity_failure_reason", "場景身份或道路尺度不符")
        override_reasons.append(f"[P3 場景身份一票否決] {id_reason}")
        zone_verification_map[TopologicalZoneType.STREET_CORRIDOR.value] = False
        if TopologicalZoneType.STREET_CORRIDOR.value not in failed_zones:
            failed_zones.insert(0, TopologicalZoneType.STREET_CORRIDOR.value)

    pass_t1 = (
        is_photorealistic
        and daylight_ok
        and bool(t1.get("perspective_and_corridor_aligned", False))
        and bool(t1.get("envelope_and_sky_compliant", False))
        and bool(t1.get("render_integrity_and_color_valid", False))
    )
    pass_t2 = (
        vehicles_realistic
        and pedestrians_authentic
        and bool(t2.get("architectural_typology_verified", False))
        and bool(t2.get("urban_density_and_commercial_fidelity", False))
        and bool(t2.get("signage_and_vehicles_valid", False))
    )
    all_zones_passed = all(zone_verification_map.values())

    overall_pass = pass_identity and pass_t1 and pass_t2 and all_zones_passed

    s_id["pass_scene_identity"] = pass_identity
    t1["pass_track1"] = pass_t1
    t2["pass_track2"] = pass_t2
    qa_dict["overall_pass"] = overall_pass
    qa_dict["failed_topological_zones"] = failed_zones

    sub_checks = [
        is_photorealistic,
        is_full_color,
        daylight_ok,
        vehicles_realistic,
        pedestrians_authentic,
        bool(s_id.get("road_scale_and_width_preserved", False)),
        bool(s_id.get("regional_urban_identity_verified", False)),
        bool(s_id.get("no_village_or_alley_degradation", False)),
        bool(t1.get("perspective_and_corridor_aligned", False)),
        bool(t1.get("envelope_and_sky_compliant", False)),
        bool(t1.get("render_integrity_and_color_valid", False)),
        bool(t2.get("architectural_typology_verified", False)),
        bool(t2.get("urban_density_and_commercial_fidelity", False)),
        bool(t2.get("signage_and_vehicles_valid", False)),
    ]
    raw_score = round(
        (sum(1 for chk in sub_checks if chk) / float(len(sub_checks)))
        * PERCENTAGE_SCALE,
        2,
    )
    qa_compliance_score = (
        raw_score if overall_pass else min(MAX_SCORE_ON_VETO, raw_score)
    )

    primary_defect = (
        "NONE"
        if overall_pass
        else (
            failed_zones[0]
            if failed_zones
            else TopologicalZoneType.CENTRAL_BACKDROP.value
        )
    )
    qa_dict["primary_defect_zone"] = primary_defect

    passing_zone_count = sum(1 for v in zone_verification_map.values() if v)
    has_valid_global_foundation = (
        is_photorealistic
        and daylight_ok
        and (not haze_washout)
        and pass_identity
        and passing_zone_count >= MIN_FOUNDATION_PASSING_ZONES
    )

    evidence_bboxes = [
        _sanitize_evidence_bbox_1000(inv.get("tallest_building_bbox_xyxy")),
        _sanitize_evidence_bbox_1000(inv.get("visible_sky_bbox_xyxy")),
        _sanitize_evidence_bbox_1000(inv.get("primary_vehicle_bbox_xyxy")),
    ]

    error_level = "NONE" if overall_pass else "CONTENT_REJECT"
    reasoning_list = qa_dict.get("reasoning_steps", []) + override_reasons
    reasoning_str = " | ".join(reasoning_list)

    merged_semantic_data = (
        dict(plan_data) if isinstance(plan_data, dict) else {}
    )
    merged_semantic_data.pop("color_reference_bytes", None)
    merged_semantic_data.pop("reference_image_bytes", None)
    merged_semantic_data["plan_id"] = plan_id
    merged_semantic_data["historical_reference_applied"] = (
        historical_reference_applied
    )
    merged_semantic_data["pass_validation"] = overall_pass
    merged_semantic_data["error_level"] = error_level
    merged_semantic_data["zone_verification_map"] = zone_verification_map
    merged_semantic_data["dual_track_qa"] = qa_dict

    verdict_label = (
        "✅ PASS"
        if overall_pass
        else f"❌ REJECT (QA得分={qa_compliance_score} | 寫實全彩={is_photorealistic} | 車輛合規={vehicles_realistic} | 行人合規={pedestrians_authentic} | 未達標分區={failed_zones})"
    )
    logger.info(
        f"⚖️ [Stage 4 三圖對照取證結果] Plan: {plan_id} (QA附歷史圖={has_ref_image} | 生成器IPAdapter實證生效={historical_reference_applied}) | "
        f"P3場景身份: {pass_identity} | Track1幾何畫質: {pass_t1} | Track2史實密度: {pass_t2} | "
        f"實測最高樓層: {estimated_stories}層 (建築高度佔比={measured_height:.2f}) | 最終判定: {verdict_label}"
    )

    return {
        "pass": overall_pass,
        "qa_compliance_score": qa_compliance_score,
        "passing_zone_count": passing_zone_count,
        "pass_scene_identity": pass_identity,
        "is_photorealistic": is_photorealistic,
        "daylight_ok": daylight_ok,
        "vehicles_realistic": vehicles_realistic,
        "pedestrians_authentic": pedestrians_authentic,
        "historical_reference_applied": historical_reference_applied,
        "measured_building_height_ratio": measured_height,
        "estimated_max_building_stories": estimated_stories,
        "has_valid_global_foundation": has_valid_global_foundation,
        "zone_verification_map": zone_verification_map,
        "reason": reasoning_str,
        "bboxes": evidence_bboxes,
        "error_level": error_level,
        "primary_defect_zone": primary_defect,
        "failed_region_ids": qa_dict.get("failed_region_ids", []),
        "failed_topological_zones": failed_zones,
        "raw_semantic_data": merged_semantic_data,
    }


# ==============================================================================
# 9. 跨檔案相容橋接器
# ==============================================================================
async def evaluate_image(
    image_bytes: bytes,
    year: str,
    location: str,
    original_image_bytes: Optional[bytes] = None,
    plan_data: Optional[Dict[str, Any]] = None,
    reference_image_bytes: Optional[bytes] = None,
    historical_reference_applied: bool = False,
) -> Dict[str, Any]:
    if original_image_bytes is not None:
        return await verify_reconstruction_dual_track(
            original_image_bytes=original_image_bytes,
            generated_image_bytes=image_bytes,
            year=year,
            location=location,
            plan_data=plan_data,
            reference_image_bytes=reference_image_bytes,
            historical_reference_applied=historical_reference_applied,
        )
    return await plan_historical_reconstruction(
        image_bytes=image_bytes, year=year, location=location
    )