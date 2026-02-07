from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date
from typing import (Annotated, Dict, Literal, Optional, Union)
from pydantic import BaseModel, ConfigDict

NumberOrNull = Optional[Union[int, float]]
ColType = Literal["str", "int", "float"]

class FinancialEntities(BaseModel):
    Company: Optional[List[str]] = None
    Date: Optional[List[str]] = None
    Location: Optional[List[str]] = None
    Money: Optional[List[str]] = None
    Person: Optional[List[str]] = None
    Product: Optional[List[str]] = None
    Quantity: Optional[List[str]] = None




class ClaimHeader(BaseModel):
    claim_id: str = Field(
        ..., description="Claim ID in format CLM-XXXXXX, where X is a digit"
    )
    report_date: date = Field(..., description="Date claim was reported")
    incident_date: date = Field(..., description="Date incident occurred")
    reported_by: str = Field(
        ..., min_length=1, description="Full name of person reporting claim"
    )
    channel: Literal["Email", "Phone", "Portal", "In-Person"] = Field(
        ..., description="Channel used to report claim"
    )

    class Config:
        extra = "forbid"


class PolicyDetails(BaseModel):
    policy_number: str = Field(
        ..., description="Policy number in format POL-XXXXXXXXX, where X is a digit"
    )
    policyholder_name: str = Field(
        ..., min_length=1, description="Full legal name on policy"
    )
    coverage_type: Literal[
        "Property", "Auto", "Liability", "Health", "Travel", "Other"
    ] = Field(..., description="Type of insurance coverage")
    effective_date: date = Field(..., description="Policy effective start date")
    expiration_date: date = Field(..., description="Policy expiration end date")

    class Config:
        extra = "forbid"


class InsuredObject(BaseModel):
    object_id: str = Field(
        ...,
        description="Unique identifier for insured object. For vehicles, use VIN format (e.g., VIN12345678901234567). For buildings, use PROP-XXXXXX format. For liability, use LIAB-XXXXXX format. For other objects, use OBJ-XXXXXX format, where X is a digit",
    )
    object_type: Literal["Vehicle", "Building", "Person", "Other"] = Field(
        ..., description="Type of insured object"
    )
    make_model: Optional[str] = Field(
        None,
        description="Make and model for vehicles (use standardtized manufacturer names and models), or building type for property",
    )
    year: Optional[int] = Field(
        None, description="Year for vehicles or year built for buildings"
    )
    location_address: Optional[str] = Field(
        None,
        description="Full street address where object is located or originated from",
    )
    estimated_value: Optional[int] = Field(
        None, description="Estimated monetary value in USD without currency symbol"
    )

    class Config:
        extra = "forbid"


class IncidentDescription(BaseModel):
    incident_type: Literal[
        "rear_end_collision",
        "side_impact_collision",
        "head_on_collision",
        "parking_lot_collision",
        "house_fire",
        "kitchen_fire",
        "electrical_fire",
        "burst_pipe_flood",
        "storm_damage",
        "roof_leak",
        "slip_and_fall",
        "property_injury",
        "product_liability",
        "theft_burglary",
        "vandalism",
    ] = Field(..., description="Specific standardized incident type")

    location_type: Literal[
        "intersection",
        "highway",
        "parking_lot",
        "driveway",
        "residential_street",
        "residence_interior",
        "residence_exterior",
        "commercial_property",
        "public_property",
    ] = Field(..., description="Standardized location type where incident occurred")

    estimated_damage_amount: Optional[int] = Field(
        None, description="Estimated damage in USD without currency symbol"
    )
    police_report_number: Optional[str] = Field(
        None, description="Police report number if applicable"
    )

    class Config:
        extra = "forbid"


class InsuranceClaim(BaseModel):
    header: ClaimHeader = Field(..., description="Basic claim information")
    policy_details: Optional[PolicyDetails] = Field(
        None, description="Policy information if available"
    )
    insured_objects: Optional[List[InsuredObject]] = Field(
        None, description="List of insured objects involved, if applicable"
    )
    incident_description: IncidentDescription = Field(
        ..., description="Structured incident details"
    )

    class Config:
        extra = "forbid"



data_table_extraction_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "StructuredSynthetic",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "num_rows": {"type": "integer"},
                "num_columns": {"type": "integer"},
                "column_types": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["str", "int", "float"],
                    },
                },
                "column_max": {
                    "type": "object",
                    "additionalProperties": {"type": ["number", "null"]},
                },
                "column_min": {
                    "type": "object",
                    "additionalProperties": {"type": ["number", "null"]},
                },
                "identifier_first": {"type": ["string", "null"]},
                "identifier_last": {"type": ["string", "null"]},
                "identifier_shortest": {"type": ["string", "null"]},
            },
            "required": [
                "num_rows",
                "num_columns",
                "column_types",
                "column_max",
                "column_min",
                "identifier_first",
                "identifier_last",
                "identifier_shortest",
            ],
        },
    },
}





class DataTableExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")  # additionalProperties: false

    num_rows: int
    num_columns: int

    # additionalProperties: {"type":"string","enum":["str","int","float"]}
    column_types: Dict[str, ColType]

    # additionalProperties: {"type":["number","null"]}
    column_max: Dict[str, NumberOrNull]
    column_min: Dict[str, NumberOrNull]

    identifier_first: Optional[str]
    identifier_last: Optional[str]
    identifier_shortest: Optional[str]




class PII(BaseModel):
    ACCOUNTNAME: Optional[str] = None
    ACCOUNTNUMBER: Optional[str] = None
    AGE: Optional[str] = None
    AMOUNT: Optional[str] = None
    BIC: Optional[str] = None
    BITCOINADDRESS: Optional[str] = None
    BUILDINGNUMBER: Optional[str] = None
    CITY: Optional[str] = None
    COMPANYNAME: Optional[str] = None
    COUNTY: Optional[str] = None
    CREDITCARDCVV: Optional[str] = None
    CREDITCARDISSUER: Optional[str] = None
    CREDITCARDNUMBER: Optional[str] = None
    CURRENCY: Optional[str] = None
    CURRENCYCODE: Optional[str] = None
    CURRENCYNAME: Optional[str] = None
    CURRENCYSYMBOL: Optional[str] = None
    DATE: Optional[str] = None
    DOB: Optional[str] = None
    EMAIL: Optional[str] = None
    ETHEREUMADDRESS: Optional[str] = None
    EYECOLOR: Optional[str] = None
    FIRSTNAME: Optional[str] = None
    GENDER: Optional[str] = None
    HEIGHT: Optional[str] = None
    IBAN: Optional[str] = None
    IP: Optional[str] = None
    IPV4: Optional[str] = None
    IPV6: Optional[str] = None
    JOBAREA: Optional[str] = None
    JOBTITLE: Optional[str] = None
    JOBTYPE: Optional[str] = None
    LASTNAME: Optional[str] = None
    LITECOINADDRESS: Optional[str] = None
    MAC: Optional[str] = None
    MASKEDNUMBER: Optional[str] = None
    MIDDLENAME: Optional[str] = None
    NEARBYGPSCOORDINATE: Optional[str] = None
    ORDINALDIRECTION: Optional[str] = None
    PASSWORD: Optional[str] = None
    PHONEIMEI: Optional[str] = None
    PHONENUMBER: Optional[str] = None
    PIN: Optional[str] = None
    PREFIX: Optional[str] = None
    SECONDARYADDRESS: Optional[str] = None
    SEX: Optional[str] = None
    SSN: Optional[str] = None
    STATE: Optional[str] = None
    STREET: Optional[str] = None
    TIME: Optional[str] = None
    URL: Optional[str] = None
    USERAGENT: Optional[str] = None
    USERNAME: Optional[str] = None
    VEHICLEVIN: Optional[str] = None
    VEHICLEVRM: Optional[str] = None
    ZIPCODE: Optional[str] = None



class GSM8KSchema(BaseModel):
    steps: Annotated[List[str], Field(min_length=1)]
    answer: Annotated[str, Field(min_length=1)]