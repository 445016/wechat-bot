import time

import openai
from zhipuai import ZhipuAI
from bot.bot import Bot
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf
from bot.zhipuai.zhipu_ai_image import ZhipuAIImage
from bot.zhipuai.zhipu_ai_session import ZhipuAISession
from bot.zhipuai.zhipu_ai_image_vision import ZhipuAIImageVision


class ZhipuAIBot(Bot, ZhipuAIImage, ZhipuAIImageVision):
    def __init__(self):
        super().__init__()
        # ZhipuAI.api_key = conf().get("zhipuai_api_key")
        # if conf().get("zhipuai_api_key"):
        #     ZhipuAI.base_url = conf().get("zhipu_ai_api_base")
        # proxy = conf().get("proxy")
        # if proxy:
        #     openai.proxy = proxy

        self.sessions = SessionManager(ZhipuAISession, model=conf().get("model") or "glm-4")
        self.args = {
            "model": conf().get("model") or "glm-4",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.9),  # 值在[0,1]之间，越大表示回复越具有不确定性
            "max_tokens": 1200,  # 回复最大的字符数
            "top_p": 0.7,
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
            "stop": ["\n\n\n"],
        }

    def reply(self, query, context=None):
        # acquire reply content
        if context and context.type:
            if context.type == ContextType.TEXT:
                logger.info("[ZHIPU_AI] query={}".format(query))
                session_id = context["session_id"]
                reply = None
                if query == "#清除记忆":
                    self.sessions.clear_session(session_id)
                    reply = Reply(ReplyType.INFO, "记忆已清除")
                elif query == "#清除所有":
                    self.sessions.clear_all_session()
                    reply = Reply(ReplyType.INFO, "所有人记忆已清除")
                else:
                    session = self.sessions.session_query(query, session_id)
                    result = self.reply_text(session)
                    total_tokens, completion_tokens, reply_content = (
                        result["total_tokens"],
                        result["completion_tokens"],
                        result["content"],
                    )
                    logger.debug(
                        "[ZHIPU_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                            str(session), session_id, reply_content, completion_tokens)
                    )

                    if total_tokens == 0:
                        reply = Reply(ReplyType.ERROR, reply_content)
                    else:
                        self.sessions.session_reply(reply_content, session_id, total_tokens)
                        reply = Reply(ReplyType.TEXT, reply_content)
                return reply
            elif context.type == ContextType.IMAGE_CREATE:
                ok, retstring = self.create_img(query=query)
                reply = None
                if ok:
                    reply = Reply(ReplyType.IMAGE_URL, retstring)
                else:
                    reply = Reply(ReplyType.ERROR, retstring)
                return reply

            elif context.type == ContextType.IMAGE:
                ok, result = self.vision_img(query=query)
                session_id = context["session_id"]
                session = self.sessions.session_query(query, session_id)
                reply = None
                if ok:
                    message = result.choices[0].message.content
                    res_content = message.strip().replace("<|endoftext|>", "")
                    total_tokens = result.usage.total_tokens
                    completion_tokens = result.usage.completion_tokens
                    logger.debug(
                        "[ZHIPU_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                            str(session), session_id, result, completion_tokens)
                    )
                    self.sessions.session_reply(message, session_id, total_tokens)
                    reply = Reply(ReplyType.TEXT, res_content)
                else:
                    reply = Reply(ReplyType.ERROR, result)
                return reply

    def reply_text(self, session: ZhipuAISession, retry_count=0):
        try:
            zhipuClient = ZhipuAI(api_key=conf().get("zhipuai_api_key"))
            response = zhipuClient.chat.completions.create(
                messages=session.messages,
                **self.args
            )

            res_content = response.choices[0].message.content.strip().replace("<|endoftext|>", "")
            total_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
            logger.info("[ZHIPU_AI] reply={}".format(res_content))
            return {
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "content": res_content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[ZHIPU_AI] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[ZHIPU_AI] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[ZHIPU_AI] APIConnectionError: {}".format(e))
                need_retry = False
                result["content"] = "我连接不到你的网络"
            else:
                logger.warn("[ZHIPU_AI] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[ZHIPU_AI] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, retry_count + 1)
            else:
                return result
