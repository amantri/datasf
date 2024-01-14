import pandas as pd
import json


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

instruct_template = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{complaint} [/INST] {model_answer} </s>"""


def main():
    # csv = pd.read_json('data-311.json')
    # df = pd.read_csv('311_Cases_20240113_10k.csv')

    data = []

    # Open the JSONL file
    with open("parsed_complaints.jsonl", "r") as file:
        for line in file:
            # Parse the JSON data from each line
            json_data = json.loads(line)
            # Add the data to your list
            data.append(json_data)

    df1 = pd.read_csv("complaints.csv")
    # df2 = pd.read_json('parsed_complaints.jsonl', lines=True)

    ft_data = []
    with open("ft_data.jsonl", "w") as file:
        for i, d in enumerate(data):
            d.pop("description", None)
            f"{instruct_template}"
            json_obj = {
                "text": instruct_template.format(
                    system_prompt=system_prompt,
                    complaint=df1.loc[i]["complaint"],
                    model_answer=json.dumps(d),
                )
            }
            ft_data.append(json_obj)
            # print(d['complaint'], d['category'])
            file.write(json.dumps(json_obj) + "\n")

    i = 0


if __name__ == "__main__":
    main()


