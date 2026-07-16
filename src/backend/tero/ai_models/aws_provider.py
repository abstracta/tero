import io
import json
import logging
import time
from typing import Optional, cast
import urllib.request
import uuid

import asyncio
import boto3
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
import tiktoken

from ..core.env import env
from .domain import AiModelProvider


logger = logging.getLogger(__name__)


def build_aws_service_client(service_name: str):
    return boto3.client(service_name, region_name=env.aws_region, **build_aws_creds())


def build_aws_creds() -> dict:
    return {
        "aws_access_key_id": env.aws_access_key_id.get_secret_value(),
        "aws_secret_access_key": env.aws_secret_access_key.get_secret_value()
    } if env.aws_access_key_id and env.aws_secret_access_key else {}


class AWSProvider(AiModelProvider):

    def __init__(self):
        super().__init__()
        self.model_arn_map = {}

    def _build_chat_model(self, model: str, temperature: Optional[float], reasoning_effort: Optional[str], streaming: bool) -> BaseChatModel:
        aws_model_id = env.aws_model_id_mapping.get(model)
        if not aws_model_id:
            raise ValueError(f"Model {model} not supported by AWS")
        return ChatBedrockConverse(
            region_name=env.aws_region,
            model=self._get_model_arn(aws_model_id),
            provider=self._get_model_provider(aws_model_id),
            temperature=temperature,
            **build_aws_creds())

    def supports_model(self, model: str) -> bool:
        return model in env.aws_model_id_mapping

    def _get_model_arn(self, model: str) -> str:
        if model not in self.model_arn_map:
            if not env.aws_region:
                raise ValueError("AWS region is not set")
            bedrock_client = build_aws_service_client("bedrock")
            inference_profiles = bedrock_client.list_inference_profiles()
            inference_profile_summaries = inference_profiles['inferenceProfileSummaries']
            for inference_profile in inference_profile_summaries:
                profile_id = inference_profile["inferenceProfileId"]
                model_id = profile_id.replace("us.", "").replace("eu.", "")
                if model_id == model:
                    arn = inference_profile["inferenceProfileArn"]
                    self.model_arn_map[model] = arn
                    return arn
            raise ValueError(f"Model {model} not supported")
        return self.model_arn_map[model]

    def _get_model_provider(self, model: str) -> str:
        return model.split(".")[0]

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return isinstance(exc, ClientError) and exc.response['Error']['Code'] in (
            'ThrottlingException', 'TooManyRequestsException'
        )

    def build_embedding(self, model: str) -> BedrockEmbeddings:
        return BedrockEmbeddings(
            region_name=env.aws_region,
            model_id=cast(str, env.aws_model_id_mapping.get(model)),
            **build_aws_creds())

    def count_tokens(self, txt: str, model: str) -> int:
        # using cl100k_base for the time being since is a good approximation to different models and
        # as long as docs chunk size is significant lower than embedding models context limit,
        # the difference between actual model token count and this count should not affect the final result significantly
        return len(tiktoken.get_encoding("cl100k_base").encode(txt))

    async def transcribe_audio(self, file: io.BytesIO, model: str) -> str:
        s3 = build_aws_service_client("s3")
        transcription_key = f"transcribe/{uuid.uuid4()}.webm"
        # Upload to S3 since AWS Transcribe requires S3 URI for async jobs.
        await self._run_blocking(
            s3.put_object,
            Bucket=env.aws_s3_bucket,
            Key=transcription_key,
            Body=file.read(),
            ContentType="audio/webm")
        try:
            return await self._transcribe_bucket_file(transcription_key)
        finally:
            try:
                await self._run_blocking(
                    s3.delete_object,
                    Bucket=env.aws_s3_bucket,
                    Key=transcription_key)
            except Exception as e:
                logger.warning(f"Failed to delete transcription file {transcription_key}", exc_info=True)

    async def _run_blocking(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _transcribe_bucket_file(self, transcription_key: str) -> str:
        transcribe = build_aws_service_client("transcribe")
        job_name = f"tero-{uuid.uuid4()}"
        await self._run_blocking(
            transcribe.start_transcription_job,
            TranscriptionJobName=job_name,
            IdentifyLanguage=True,
            Media={"MediaFileUri": f"s3://{env.aws_s3_bucket}/{transcription_key}"},
            MediaFormat="webm")

        transcription_timeout_seconds = 120
        deadline = time.time() + transcription_timeout_seconds
        while time.time() < deadline:
            job = await self._run_blocking(
                transcribe.get_transcription_job,
                TranscriptionJobName=job_name)
            status = job["TranscriptionJob"]["TranscriptionJobStatus"]
            if status == "COMPLETED":
                transcript_uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                response_body = await self._run_blocking(urllib.request.urlopen, transcript_uri)
                with response_body as resp:
                    data = json.loads(resp.read())
                    return data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
            elif status == "FAILED":
                failure_reason = job["TranscriptionJob"].get("FailureReason", "Unknown error")
                raise RuntimeError(f"Transcription job {job_name} failed: {failure_reason}")
            await asyncio.sleep(1)
        try:
            await self._run_blocking(
                transcribe.delete_transcription_job,
                TranscriptionJobName=job_name)
        except Exception:
            logger.warning(f"Failed to delete transcription file {transcription_key}", exc_info=True)
        raise RuntimeError(f"Transcription timed out after {transcription_timeout_seconds} seconds")
