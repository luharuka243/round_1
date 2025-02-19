from pydantic import BaseModel, Field

class InputValidator(BaseModel):
    """
    Input validator for the prediction API
    """
    complaint_id: str = Field("Not present", description="Complaint ID is a required string")
    content: str = Field(..., description="Content is a required string")

class OutputValidator(BaseModel):
    """
    Output validator for the prediction API
    """
    category: str = Field(..., description="Category is a required string")
    sub_category: str = Field(..., description="Sub category is a required string")
