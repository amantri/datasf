from typing import Union
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import together
import json

together.api_key = "efc033cc14ee56d571a653fa044e737db53b35279fc9c217bb8cee7283997315"

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class Data(BaseModel):
    complaint: str
    image: Optional[str] = None

def get_prompt(complaint):
    system_prompt = """
You are a helpful assistant that helps people file their complaints to the city.

Parse the given complaint into the json schema specified below. Only output the fields that can be parsed. Do not output anything other than json

<json_schema>
{
  "address_string": "Human entered address or description of location. This is required if no lat/long or address_id are provided. This should be written from most specific to most general geographic unit, eg address number or cross streets, street name, neighborhood/district, city/town/village, county, postal code.",
  "email": "The email address of the person submitting the request",
  "first_name": "The given name of the person submitting the request",
  "last_name": "The family name of the person submitting the request",
  "phone": "The phone number of the person submitting the request",
}
</json_schema>
"""

#     instruct_template = """<s>[INST] <<SYS>>
# {system_prompt}
# <</SYS>>

# {complaint} [/INST] </s>"""

    instruct_template = """
{system_prompt}

{complaint}"""

    return instruct_template.format(
                    system_prompt=system_prompt,
                    complaint=complaint)


@app.post(f"/complaint")
def parse_complaint(data: Data):
    print(f"{data=}")
    prompt = get_prompt(data.complaint)
    print(f"{prompt=}\n\n")
    output = together.Complete.create(
        prompt=prompt,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        # model="anup.mantri@gmail.com/Mistral-7B-Instruct-v0.2-2024-01-14-02-51-03",
        max_tokens=256,
        temperature=0.8,
        top_k=60,
        top_p=0.6,
        repetition_penalty=1.1,
    )

    ret = output["output"]["choices"][0]["text"]
    print(f'{ret=}')
    
    # Reflect the image back as the media url
    # ret["media_url"] = data.image

    return ret
