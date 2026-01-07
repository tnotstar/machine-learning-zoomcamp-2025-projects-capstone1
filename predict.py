import pickle
import pprint as pp

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class VariantRequest(BaseModel):
    chrom: str = "1"
    pos: int =   1168180
    ref: str =   "G"
    alt: str =   "C"
    af_esp: float = 0.07710
    mc: str =    "SO:0001583|missense_variant"
    origin: int = 1
    protein_position: float = 174.0
    amino_acids: str = "E/D"
    strand: str = "1"
    loftool: float = 0.157
    cadd_phred: float = 1.053
    cadd_raw: float = -0.208682
    location_type: str = "exon"
    location_value: str = "1/1"


class PredictionResponse(BaseModel):
    conflict_probability: float
    is_conflict: bool


app = FastAPI(title="Conflicting Classifications Prediction Service")

with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict_single(variant) -> float:
    X = dv.transform([variant])
    result = model.predict_proba(X)[0, 1]
    return float(result)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post("/predict")
async def predict(variant: VariantRequest) -> PredictionResponse:
    pp.pprint(variant)
    tx = variant.model_dump()
    pp.pprint(tx)
    prob = predict_single(tx)
    return PredictionResponse(conflict_probability=prob, is_conflict=(prob >= 0.5))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
