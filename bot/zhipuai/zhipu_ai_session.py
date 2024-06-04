from bot.session_manager import Session
from common.log import logger


class ZhipuAISession(Session):
    def __init__(self, session_id, system_prompt=None, model="text-davinci-003"):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.reset()

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        precise = True
        try:
            cur_tokens = self.calc_tokens()
        except Exception as e:
            precise = False
            if cur_tokens is None:
                raise e
            logger.debug("Exception when counting tokens precisely for query: {}".format(e))
        while cur_tokens > max_tokens:
            if len(self.messages) > 1:
                self.messages.pop(0)
            elif len(self.messages) == 1 and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)
                if precise:
                    cur_tokens = self.calc_tokens()
                else:
                    cur_tokens = len(str(self))
                break
            elif len(self.messages) == 1 and self.messages[0]["role"] == "user":
                logger.warn("user question exceed max_tokens. total_tokens={}".format(cur_tokens))
                break
            else:
                logger.debug("max_tokens={}, total_tokens={}, len(conversation)={}".format(max_tokens, cur_tokens, len(self.messages)))
                break
            if precise:
                cur_tokens = self.calc_tokens()
            else:
                cur_tokens = len(str(self))
        return cur_tokens

    def calc_tokens(self):
        return num_tokens_from_string(self)


#Token是模型用来表示自然语言文本的基本单位，可以直观的理解为“字”或“词”；通常1个中文词语、1个英文单词、1个数字或1个符号计为 1 个token。
def num_tokens_from_string(self) -> int:
    """Returns the number of tokens in a text string."""
    tokens = 0
    for msg in self.messages:
        # 官方token计算规则暂不明确： "大约为 token数为 "中文字 + 其他语种单词数 x 1.3"
        # 这里先直接根据字数粗略估算吧，暂不影响正常使用，仅在判断是否丢弃历史会话的时候会有偏差
        tokens += len(msg["content"])
    return tokens
