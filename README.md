# :parrot: :hugs: :robot: Langchain HuggingGPT

A lightweight implementation of [HuggingGPT](https://arxiv.org/abs/2303.17580) with [langchain](https://docs.langchain.com/docs/). 
No local inference, only models available on the [huggingface inference API](https://huggingface.co/inference-api) are used. 

## Getting Started

Install the package with pip:

```commandline
pip install -r requirements.txt
pip install -e .
```

or with your preferred virtual environment manager (_this project uses [pdm](https://pdm.fming.dev/) for dependency management_).

Setup your OpenAI and Huggingface Hub credentials:

```commandline
cp .env.example .env
```

Then fill in the `.env` file with your `OPENAI_API_KEY` & `HUGGINGFACEHUB_API_TOKEN` credentials.

## Usage

```commandline
python main.py
```

Then converse with HuggingGPT, e.g:

```commandline
Please enter your request. End the conversation with 'exit'
User: Draw me a sheep
Assistant:
I have carefully considered your request and based on the inference results, I have generated an image of a sheep for you. The image can be found at the following URL: /images/1e34.png. The model I used for this task was runwayml/stable-diffusion-v1-5, as it is specifically designed for text-to-image tasks and has the highest number of likes. I hope this image meets your expectations. Is there anything else I can help you with?
User: exit
```

<img src="resources/images/1e34.png" alt="sheep drawing" width="300"/>

To use the application in standalone mode, use the `--prompt` flag:

```commandline
python main.py --prompt "Draw me a sheep"
```

## Examples


### Image Generation + Image-to-Text + Object Detection + Visual Question Answering

```commandline
Please enter your request. End the conversation with 'exit'
User: Generate an image of a field with three sheep grazing peacefully        
Assistant:
I have carefully considered your request and based on the inference results, I have generated an image of a field with three sheep grazing peacefully. The model I used for this task is runwayml/stable-diffusion-v1-5, which is the most suitable for the task as it is the most popular model with the highest number of likes (6367) and it has the most relevant tags (stable-diffusion, stable-diffusion-diffusers, text-to-image) for the task. The complete path or url of the generated image is /images/1e70.png. Is there anything else I can help you with?
User: How many sheep are in the image?
Assistant:
Based on the inference results, there are 4 sheep in the image. I used the following models and inference results to answer your request: 

Model 1: nlpconnect/vit-gpt2-image-captioning. This model is trained by @ydshieh and is specifically designed for image-to-text tasks. It generated the text "sheep grazing in a field" from the image. 

Model 2: facebook/detr-resnet-50. This model is suitable for the task of object-detection as it is specifically designed for this purpose. It has a ResNet-50 backbone which is a powerful feature extractor and is capable of detecting objects in an image. It also has a local inference endpoint which makes it faster and more stable. It generated the image with predicted boxes "/images/c6fd.png" and predicted 4 sheep in the image. 

Model 3: dandelin/vilt-b32-finetuned-vqa. This model is a Vision-and-Language Transformer (ViLT) that has been fine-tuned on VQAv2. It is specifically designed for visual-question-answering tasks, which
User: exit
```

<img src="resources/images/1e70.png" alt="sheep" width="300"/>
<img src="resources/images/c6fd.png" alt="sheep with predicted boxes" width="300"/>

### Text Generation + Text-to-Speech

```commandline
Please enter your request. End the conversation with 'exit'
User: Write a poem about sheep, then read it to me
Assistant:
Sure, I will do my best to answer your request. Based on the inference results, I have generated a poem about sheep for you. Here is the poem: 

Sheep in the meadow,
Grazing so sweetly,
Softly they bleat,
In the morning sun so brightly.

Wool so soft and white,
Fleece so warm and light,
A gentle reminder of nature's delight.

Sheep in the meadow,
Grazing so peacefully,
A reminder of nature's beauty,
For all of us to see. 

To generate this poem, I used the OpenAI model for text generation. The audio file of the poem can be found at the following URL: /audios/499e.flac. This audio file was generated using the ESPnet/Kan-Bayashi_LJSpeech_VITS model, which is imported from https://zenodo.org and has 70 likes. It is an ESPnet2 TTS pretrained model and is specifically designed for text-to-speech tasks.

I hope this answers your request. Is there anything else I can help you with?
User: exit
```

[sheep poem audio](resources/audios/499e.flac)

### Automatic Speech Recognition + Sentence Similarity

```commandline
Please enter your request. End the conversation with 'exit'
User: Transcribe the audio file found at /audios/499e.flac. Then tell me how similar the transcription is to the following sentence: Sheep are nice.
Assistant:
Based on the inference results, the transcription of the audio file found at /audios/499e.flac is "HERE IS YOUR POEM ABOUT SHEEP SHEEP IN THE MEADOW GRAZING SO SWEETLY SOFTLY THEY BLEAT IN THE MORNING SUN SO BRIGHTLY WOOLF SO SOFT AND WHIT FLEECE SO WARM AND LIGHT THE GENTLE REMINDER OF NATURE'S DELIGHT SHEEP IN THE MEADOW GRAZING SO PEACEFULLY A REMINDER OF NATURE'S BEAUTY FOR ALL OF US TO SEE". The similarity between the transcription and the sentence "Sheep are nice" is 0.46181225776672363. 

To complete this task, I used two models: facebook/wav2vec2-base-960h for automatic speech recognition and sentence-transformers/all-MiniLM-L6-v2 for sentence similarity. The inference results are the transcription of the audio file and the similarity score between the transcription and the sentence. The complete path or url of the audio file is /audios/499e.flac.
User: exit
```

## Future Work

- Concurrent execution of non-dependent tasks with `asyncio`
- Smarter validation of task planning
- Better support for image-to-image control tasks


## Development

### Testing

Install test dependencies with pdm:

```commandline
pdm sync
```

Run unit tests with pytest:

```commandline
pdm run pytest
```

Run smoke test:

```commandline
pdm run python tests/smoke_test.py
```


## License

[MIT license](LICENSE)

## Credits

* [JARVIS](https://github.com/microsoft/JARVIS)

