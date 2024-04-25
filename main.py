from transformers import pipeline
import os
import gradio as gr
from openai import OpenAI
client = OpenAI(api_key = '')
# Use environment variables for security best practices

def generate_text_with_gpt(prompt):
    try:
        # Adjusted for the chat model endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Specify the appropriate chat model
            messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                      {"role": "user", "content": f'pls help me check the price range of{prompt}'}],
        )
        # Adjust response parsing for the chat completions format
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

app2 = gr.Interface(
    fn=generate_text_with_gpt,
    inputs=gr.Textbox(lines=1, placeholder="Enter name of the shoes you wanna check"),
    outputs="text",
    title="Nike Shoes classification",
)
pipe = pipeline(task="image-classification",
                model="HZhang729/nike_image_classification")

app1 = gr.Interface.from_pipeline(pipe,
                                  title="Shoes Recognition",
                                  description="Please upload a picture of Nike shoes, and the model will help you recognize it",

                                  )

demo = gr.TabbedInterface(
    [app1, app2],
    tab_names=["Shoes recognizer", "Price"],
    title="Nike Shoes price check"
)

demo.launch()