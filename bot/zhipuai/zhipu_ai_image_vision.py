import time

from zhipuai import ZhipuAI

from common.log import logger
from config import conf


class ZhipuAIImageVision(object):
    ZhipuAiClient = ZhipuAI(api_key=conf().get("zhipuai_api_key"))

    def __init__(self):
        pass

    def vision_img(self, query):
        try:
            logger.info("[ZHIPU_AI] image_vision_query={}".format(query))
            response = self.ZhipuAiClient.chat.completions.create(
                model=conf().get("image_vision") or "glm-4v",
                # messages=query
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "说出图片中的内容"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": query
                                }
                            }
                        ]
                    }
                ]
            )
            return True, response
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
