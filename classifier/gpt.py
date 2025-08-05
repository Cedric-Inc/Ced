import os
from openai import OpenAI

api_key = "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"
client = OpenAI(api_key=api_key)


PRESET_PROMPT = (
    "User Rating: R1\n"
    "Forecast: "
)

def generate_response(rating: str, forecast: str) -> str:
    full_prompt = (
        "User Rating: {rating}\n"
        "Forecast: {forecast}".format(rating=rating, forecast=forecast)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an investment advisor. Based on the user's risk level and forecast situation, you need to provide suggestions. "
                        "Your response should be within 20 words. There are three types of user levels: R1, R2, R3."
                    )
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f" {e}"


p = "There is a 39.7% probability of a stable trend."
print(p, '\n', "Based on your risk assessment:")
print(generate_response(rating="R1", forecast=p))
