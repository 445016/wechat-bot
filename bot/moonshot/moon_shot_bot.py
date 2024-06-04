# encoding:utf-8

import time

import openai
from openai import OpenAI

from bot.bot import Bot
from bot.moonshot.moon_shot_image import MoonChatImage
from bot.moonshot.moon_shot_session import MoonChatSession
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf

user_session = dict()


class MoonShotBot(Bot, MoonChatImage):

    def __init__(self):
        super().__init__()
        # openai.api_key = conf().get("moon_chat_api_key")
        # if conf().get("moon_chat_api_base"):
        #     openai.api_base = conf().get("moon_chat_api_base")
        # proxy = conf().get("proxy")
        # if proxy:
        #     openai.proxy = proxy

        self.sessions = SessionManager(MoonChatSession, model=conf().get("model") or "text-davinci-003")
        self.args = {
            "model": conf().get("model") or "moonshot-v1-128k",  # 对话模型的名称
            "temperature": 0.3,  # 值在[0,1]之间，越大表示回复越具有不确定性
            #"max_tokens": 1200,  # 回复最大的字符数
            "top_p": 1,
            "stream": False,
        }

    def reply(self, query, context=None):
        # acquire reply content
        if context and context.type:
            if context.type == ContextType.TEXT:
                logger.info("[MOON_CHAT] query={}".format(query))
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
                        "[MOON_CHAT] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(str(session), session_id, reply_content, completion_tokens)
                    )

                    if total_tokens == 0:
                        reply = Reply(ReplyType.ERROR, reply_content)
                    else:
                        self.sessions.session_reply(reply_content, session_id, total_tokens)
                        reply = Reply(ReplyType.TEXT, reply_content)
                return reply
            elif context.type == ContextType.IMAGE_CREATE:
                ok, retstring = self.create_img(query, 0)
                reply = None
                if ok:
                    reply = Reply(ReplyType.IMAGE_URL, retstring)
                else:
                    reply = Reply(ReplyType.ERROR, retstring)
                return reply


    def reply_text(self, session: MoonChatSession, retry_count=0):
        try:
            MoonChatAiClient = OpenAI(
                api_key=conf().get("moon_shot_api_key"),
                base_url=conf().get("moon_shot_api_base"),
            )
            response = MoonChatAiClient.chat.completions.create(messages=session.messages, **self.args)
            res_content = response.choices[0].message.content.strip().replace("<|endoftext|>", "")
            total_tokens = response.usage.total_tokens
            completion_tokens = response.usage.completion_tokens
            logger.info("[MOON_CHAT] reply={}".format(res_content))
            return {
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "content": res_content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.RateLimitError):
                logger.warn("[MOON_CHAT] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.APITimeoutError):
                logger.warn("[MOON_CHAT] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                logger.warn("[MOON_CHAT] APIConnectionError: {}".format(e))
                need_retry = False
                result["content"] = "我连接不到你的网络"
            else:
                logger.warn("[MOON_CHAT] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[MOON_CHAT] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, retry_count + 1)
            else:
                return result
