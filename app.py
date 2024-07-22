import streamlit as st
import requests
import io
import base64
from PIL import Image
import os
import json
import hashlib
import logging
from datetime import datetime
import asyncio
import aiohttp
from googletrans import Translator
import numpy as np

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.INFO)

# 번역기 초기화
translator = Translator()

# 다국어 지원을 위한 간단한 함수
def get_text(key, lang='ko'):
    texts = {
        'ko': {
            'title': '이미지 생성',
            'generate': '생성하기',
            'prompt': '프롬프트 입력',
            'negative_prompt': '네거티브 프롬프트 입력',
            'num_images': '이미지 수',
            'guidance_scale': '가이던스 스케일',
            'inference_steps': '추론 단계',
            'advanced_settings': '고급 설정',
            'model': '모델',
            'image_size': '이미지 크기',
            'seed': '시드',
            'generate_error': '오류 발생: {}',
            'api_error': 'API 응답: {}',
            'invalid_prompt': '유효한 프롬프트를 입력해주세요.',
            'upload_image': '이미지 선택...',
            'analyze_image': '업로드된 이미지 분석',
            'analysis_coming_soon': '이미지 분석 기능은 곧 제공됩니다!',
            'previous_images': '이전에 생성된 이미지',
            'save_image': '이미지 {} 저장',
            'image_saved': '이미지 {}가 성공적으로 저장되었습니다!',
            'rate_image': '이 이미지 평가',
            'loading': '로딩 중...',
            'error_too_many_requests': '너무 많은 요청. 잠시 후 다시 시도해주세요.',
            'inpainting_title': '인페인팅',
            'upload_edit_image': '편집할 이미지 업로드',
            'upload_mask': '마스크 이미지 업로드 (흰색 부분이 편집될 영역)',
            'original_image': '원본 이미지',
            'mask_image': '마스크',
            'inpaint_prompt': '인페인팅 프롬프트',
            'inpaint_negative_prompt': '인페인팅 네거티브 프롬프트',
            'run_inpainting': '인페인팅 실행',
            'inpainting_result': '인페인팅 결과',
            'save_inpainting': '인페인팅 결과 저장',
            'inpainting_saved': '인페인팅 결과가 저장되었습니다!',
            'inpainting_error': '인페인팅 중 오류 발생: {}',
            'invalid_inpaint_prompt': '유효한 인페인팅 프롬프트를 입력해주세요.'
        },
    }
    return texts.get(lang, texts['ko']).get(key, key)

# Stability AI API 설정
API_HOST = 'https://api.stability.ai'
API_KEY = st.secrets.get("STABILITY_API_KEY")

# API 키 유효성 검사
if not API_KEY:
    st.error("API 키가 설정되지 않았습니다. secrets에 STABILITY_API_KEY를 설정해주세요.")
    st.stop()

# 페이지 설정
st.set_page_config(page_title="이미지 생성 앱", layout="wide")

# 세션 상태 초기화
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'language' not in st.session_state:
    st.session_state.language = 'ko'
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = datetime.min

# 비동기 이미지 생성 함수
async def generate_image_async(client, url, headers, payload):
    async with client.post(url, headers=headers, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API 요청 실패 (상태 코드 {response.status}): {error_text}")
        data = await response.json()
        if "artifacts" not in data or len(data["artifacts"]) == 0:
            raise Exception("API 응답에 이미지 데이터가 없습니다")
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        return Image.open(io.BytesIO(image_data))

# 이미지 생성 함수
def generate_images(prompt, negative_prompt, num_images, guidance_scale, num_inference_steps, model, image_size, seed):
    current_time = datetime.now()
    if (current_time - st.session_state.last_request_time).total_seconds() < 1:
        raise Exception(get_text('error_too_many_requests', st.session_state.language))
    
    st.session_state.last_request_time = current_time
    
    # 프롬프트 번역
    translated_prompt = translator.translate(prompt, dest='en').text
    translated_negative_prompt = translator.translate(negative_prompt, dest='en').text if negative_prompt else ""
    
    url = f"{API_HOST}/v1/generation/{model}/text-to-image"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "text_prompts": [
            {
                "text": translated_prompt
            }
        ],
        "cfg_scale": guidance_scale,
        "height": int(image_size.split("x")[1]),
        "width": int(image_size.split("x")[0]),
        "steps": num_inference_steps,
        "samples": 1
    }
    if translated_negative_prompt:
        payload["text_prompts"].append({"text": translated_negative_prompt, "weight": -1})
    if seed != -1:
        payload["seed"] = seed

    async def generate_all_images():
        async with aiohttp.ClientSession() as client:
            tasks = [generate_image_async(client, url, headers, payload) for _ in range(num_images)]
            return await asyncio.gather(*tasks)

    return asyncio.run(generate_all_images())

# 인페인팅 함수
async def inpaint_image_async(client, url, headers, payload):
    async with client.post(url, headers=headers, json=payload) as response:
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API 요청 실패 (상태 코드 {response.status}): {error_text}")
        data = await response.json()
        if "artifacts" not in data or len(data["artifacts"]) == 0:
            raise Exception("API 응답에 이미지 데이터가 없습니다")
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        return Image.open(io.BytesIO(image_data))

def inpaint_image(image, mask, prompt, negative_prompt, guidance_scale, num_inference_steps, model, seed):
    url = f"{API_HOST}/v1/generation/{model}/image-to-image/masking"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # 이미지와 마스크를 base64로 인코딩
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    buffered = io.BytesIO()
    mask.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 프롬프트 번역
    translated_prompt = translator.translate(prompt, dest='en').text
    translated_negative_prompt = translator.translate(negative_prompt, dest='en').text if negative_prompt else ""

    payload = {
        "init_image": image_base64,
        "mask_image": mask_base64,
        "text_prompts": [
            {
                "text": translated_prompt
            }
        ],
        "cfg_scale": guidance_scale,
        "steps": num_inference_steps,
        "samples": 1
    }
    if translated_negative_prompt:
        payload["text_prompts"].append({"text": translated_negative_prompt, "weight": -1})
    if seed != -1:
        payload["seed"] = seed

    async def generate_inpainted_image():
        async with aiohttp.ClientSession() as client:
            return await inpaint_image_async(client, url, headers, payload)

    return asyncio.run(generate_inpainted_image())

# 메인 영역
st.title(get_text('title', st.session_state.language))

# 사이드바 설정
st.sidebar.title("설정")
prompt = st.sidebar.text_area(get_text('prompt', st.session_state.language), max_chars=1000)
negative_prompt = st.sidebar.text_area(get_text('negative_prompt', st.session_state.language), max_chars=1000)
num_images = st.sidebar.slider(get_text('num_images', st.session_state.language), 1, 4, 1)
guidance_scale = st.sidebar.slider(get_text('guidance_scale', st.session_state.language), 1.0, 20.0, 7.5)
num_inference_steps = st.sidebar.slider(get_text('inference_steps', st.session_state.language), 10, 150, 50)

# 고급 설정
with st.sidebar.expander(get_text('advanced_settings', st.session_state.language)):
    model = st.selectbox(get_text('model', st.session_state.language), ["stable-diffusion-xl-1024-v1-0", "stable-diffusion-v1-5"])
    image_size = st.selectbox(get_text('image_size', st.session_state.language), ["1024x1024", "512x512"])
    seed = st.number_input(get_text('seed', st.session_state.language), min_value=-1, max_value=2**32-1, value=-1)

if st.button(get_text('generate', st.session_state.language)):
    if prompt and len(prompt.strip()) > 0:
        try:
            with st.spinner(get_text('loading', st.session_state.language)):
                images = generate_images(prompt, negative_prompt, num_images, guidance_scale, num_inference_steps, model, image_size, seed)
                
                cols = st.columns(num_images)
                for idx, img in enumerate(images):
                    cols[idx].image(img, use_column_width=True)
                    if cols[idx].button(get_text('save_image', st.session_state.language).format(idx+1)):
                        img.save(f"generated_image_{idx+1}.png")
                        st.success(get_text('image_saved', st.session_state.language).format(idx+1))
                    
                    # 이미지 평가 기능
                    rating = cols[idx].slider(get_text('rate_image', st.session_state.language), 1, 5, 3, key=f"rating_{idx}")
                    if rating:
                        logging.info(f"Image {idx+1} rated {rating}/5")
                
                # 세션 상태 업데이트 (최대 20개 이미지만 저장)
                st.session_state.generated_images = (st.session_state.generated_images + images)[-20:]
        
        except Exception as e:
            st.error(get_text('generate_error', st.session_state.language).format(str(e)))
            logging.error(f"Error generating images: {str(e)}")
    else:
        st.warning(get_text('invalid_prompt', st.session_state.language))

# 인페인팅 섹션
st.title(get_text('inpainting_title', st.session_state.language))

uploaded_image = st.file_uploader(get_text('upload_edit_image', st.session_state.language), type=["png", "jpg", "jpeg"])
uploaded_mask = st.file_uploader(get_text('upload_mask', st.session_state.language), type=["png", "jpg", "jpeg"])

if uploaded_image and uploaded_mask:
    image = Image.open(uploaded_image).convert("RGB")
    mask = Image.open(uploaded_mask).convert("L")

    st.image(image, caption=get_text('original_image', st.session_state.language), use_column_width=True)
    st.image(mask, caption=get_text('mask_image', st.session_state.language), use_column_width=True)

    inpaint_prompt = st.text_area(get_text('inpaint_prompt', st.session_state.language), max_chars=1000)
    inpaint_negative_prompt = st.text_area(get_text('inpaint_negative_prompt', st.session_state.language), max_chars=1000)

    if st.button(get_text('run_inpainting', st.session_state.language)):
        if inpaint_prompt and len(inpaint_prompt.strip()) > 0:
            try:
                with st.spinner(get_text('loading', st.session_state.language)):
                    inpainted_image = inpaint_image(
                        image, mask, inpaint_prompt, inpaint_negative_prompt,
                        guidance_scale, num_inference_steps, model, seed
                    )
                    st.image(inpainted_image, caption=get_text('inpainting_result', st.session_state.language), use_column_width=True)
                    
                    # 이미지 저장 옵션
                    if st.button(get_text('save_inpainting', st.session_state.language)):
                        inpainted_image.save("inpainted_image.png")
                        st.success(get_text('inpainting_saved', st.session_state.language))
            
            except Exception as e:
                st.error(get_text('inpainting_error', st.session_state.language).format(str(e)))
                logging.error(f"Inpainting error: {str(e)}")
        else:
            st.warning(get_text('invalid_inpaint_prompt', st.session_state.language))

# 이미지 업로드 섹션
uploaded_file = st.file_uploader(get_text('upload_image', st.session_state.language), type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    if st.button(get_text('analyze_image', st.session_state.language)):
        st.info(get_text('analysis_coming_soon', st.session_state.language))

# 저장된 이미지 표시
if st.session_state.generated_images:
    st.subheader(get_text('previous_images', st.session_state.language))
    cols = st.columns(len(st.session_state.generated_images))
    for idx, img in enumerate(st.session_state.generated_images):
        cols[idx].image(img, use_column_width=True)

# 로그 기록
logging.info(f"App accessed at {datetime.now()}")