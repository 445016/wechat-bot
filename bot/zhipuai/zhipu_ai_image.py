import time

from zhipuai import ZhipuAI

from common.log import logger
from config import conf


class ZhipuAIImage(object):
    ZhipuAiClient = ZhipuAI(api_key=conf().get("zhipuai_api_key"))

    def __init__(self):
        pass

    def create_img(self, query):
        try:
            logger.info("[ZHIPU_AI] image_query={}".format(query))
            response = self.ZhipuAiClient.images.generations(
                model=conf().get("text_to_image") or "cogview-3",
                prompt=query,
            )
            image_url = response.data[0].url
            logger.info("[ZHIPU_AI] image_url={}".format(image_url))
            return True, image_url
        # except openai.error.RateLimitError as e:
        #     logger.warn(e)
        #     if retry_count < 1:
        #         time.sleep(5)
        #         logger.warn("[ZHIPU_AI] ImgCreate RateLimit exceed, 第{}次重试".format(retry_count + 1))
        #         return self.create_img(query, retry_count + 1)
        #     else:
        #         return False, "画图出现问题，请休息一下再问我吧"
        except Exception as e:
            logger.exception(e)
            return False, "画图出现问题，请休息一下再问我吧"
