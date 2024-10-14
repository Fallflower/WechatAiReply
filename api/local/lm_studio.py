from openai import OpenAI


class AIResponser:
    def __init__(self, init_ai_content, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.conn = OpenAI(base_url=base_url, api_key=api_key)

        self.history = [
            {"role": "system", "content": init_ai_content},
        ]
        # print(self.conn.models.list())

    def response(self, user_content):
        self.history.append({"role": "user", "content": user_content})

        completion = self.conn.chat.completions.create(
            model="lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
            messages= self.history,
            temperature=0.7,
            stream=True,
        )

        answer_msg = {"role": "assistant", "content": ""}

        for chunk in completion:
            resp = chunk.choices[0].delta.content
            if resp:
                answer_msg["content"] += resp
                # print(resp, end="")

        self.history.append(answer_msg)
        return answer_msg["content"]