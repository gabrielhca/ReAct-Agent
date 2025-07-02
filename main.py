import os
import re
from groq import Groq

# Certifique-se de definir a variável de ambiente 'GROQ_API_KEY' antes de executar.
# A chave da API do GROQ precisa estar definida como variável de ambiente 'GROQ_API_KEY'.

# Cria um cliente do Groq usando achave de API armazenada na variável
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Base do conhecimento - Dicionário com os principais tipos de séries
teorema_series = {
    "Séries Telescópicas": """
Definição e Característica:
  Uma série telescópica é aquela em que cada termo a_n pode ser expresso
  como a diferença de dois termos consecutivos de uma sequência:
    a_n = b_n - b_{n+1}  ou  a_n = b_{n+1} - b_n
  Característica principal: cancelamento em cascata na soma parcial:
    S_N = b_1 - b_{N+1}

Regra de Convergência:
  - Converge se o limite de b_{N+1} quando N -> infinito existe e é finito.
    Soma: b_1 - lim_{N->∞} b_{N+1}
  - Diverge caso contrário.
""",

    "Séries Geométricas": """
Definição e Característica:
  Série da forma:
    sum_{n=0}^{∞} ar^n = a + ar + ar^2 + ...
  ou
    sum_{n=1}^{∞} ar^{n-1}
  Onde:
    a = primeiro termo
    r = razão comum

Regra de Convergência:
  - Converge se |r| < 1. Soma = a / (1 - r)
  - Diverge se |r| >= 1
""",

    "Séries de Potência": """
Definição e Característica:
  Série da forma:
    sum_{n=0}^{∞} c_n (x - a)^n
  Representa uma função, com raio de convergência R.

Regra de Convergência:
  - Converge somente em x = a, se R = 0
  - Converge para todo x real, se R = infinito
  - Converge se |x - a| < R e diverge se |x - a| > R

Observação:
  Para calcular R usa-se:
    - Teste da Razão
    - Teste da Raiz
  Verifique o intervalo de convergência I nos extremos (a - R e a + R)
""",

    "Séries Alternadas": """
Definição e Característica:
  Série cujos termos alternam entre positivos e negativos, como:
    sum_{n=1}^{∞} (-1)^{n-1} b_n  ou  sum_{n=1}^{∞} (-1)^n b_n
  com b_n > 0

Regra de Convergência (Teste de Leibniz):
  A série alternada converge se:
    1. Os termos b_n são decrescentes: b_{n+1} <= b_n (a partir de algum n)
    2. lim_{n->∞} b_n = 0

Convergência Absoluta vs Condicional:
  - Convergência Absoluta: sum |a_n| converge
  - Convergência Condicional: sum a_n converge, mas sum |a_n| diverge
""",

    "Série Harmônica": """
Definição e Característica:
  Série harmônica:
    sum_{n=1}^{∞} 1/n = 1 + 1/2 + 1/3 + 1/4 + ...
  Mesmo que os termos tendam a zero, a série diverge.

Regra de Divergência:
  - A série harmônica diverge.
""",

    "Série de Taylor": """
Definição e Característica:
  Série de Taylor de f(x), centrada em x = a:
    f(x) = sum_{n=0}^{∞} (f^n(a)/n!) * (x - a)^n

Regra de Convergência:
  - A série converge para f(x) se o resto de Taylor tende a zero:
      lim_{n->∞} R_n(x) = 0, com R_n(x) = f(x) - P_n(x)
  - Pode-se usar a estimativa:
      Se |f^{n+1}(t)| <= M, então:
        |R_n(x)| <= M * |x - a|^{n+1} / (n+1)!
  - Se o limite de R_n(x) tende a 0, a série converge para f(x).
"""
}

# Função que busca no dicionário o conteúdo de acordo com o nome da série fornecido
# A comparação é feita em lowercase
def get_convergence_rule(series_name: str) -> str:
    cleaned_series_name = series_name.strip().lower()
    for key in teorema_series:
        if key.lower() == cleaned_series_name:
            return f"**{key}**\n\n{teorema_series[key]}"

    return f"Regra ou teorema para '{series_name}' não encontrado no documento."

# Função que simula o agente inteligente e mantém um historico de conversa
class Agent:
    def __init__(self, client: Groq, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = self.client.chat.completions.create(
            model="llama3-70b-8192", messages=self.messages
        )
        return completion.choices[0].message.content

# Definição do protocolo de raciocínio que o modelo deve seguir
system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available action is:

get_convergence_rule:
e.g. get_convergence_rule: Série Harmônica
Returns the definition, characteristics, and convergence/divergence rules for the specified series based on the provided document.

Example session:

Question: Qual é a regra para convergência de Séries Geométricas?
Thought: I need to find the rule for 'Séries Geométricas'. I will use the available tool.
Action: get_convergence_rule: Séries Geométricas
PAUSE

You will be called again with this:

Observation: **Séries Geométricas**

**Definição e Característica:** Uma série geométrica é da forma...
**Regra de Convergência/Divergência:**
* **Converge** se |r| < 1. A soma da série é a/(1-r).
* **Diverge** se |r| >= 1.

Thought: The observation contains the complete rule. I can now formulate the final answer.
Answer: Uma série geométrica converge se o módulo da sua razão 'r' for menor que 1 (|r| < 1), e diverge se o módulo da razão for maior ou igual a 1 (|r| >= 1).

Now it's your turn:
""".strip()


# Função que executa a simulação de um agente inteligente usando um loop controlado
def agent_loop(client: Groq, tools: dict, max_iterations=10, query: str = ""):
    agent = Agent(client, system_prompt)
    next_prompt = query
    i = 0
    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+):\s*(.+)", result, re.IGNORECASE)
            if action:
                chosen_tool_name = action[0][0]
                arg = action[0][1].strip()

                if chosen_tool_name in tools:
                    tool_result = tools[chosen_tool_name](arg)
                    next_prompt = f"Observation: {tool_result}"
                else:
                    next_prompt = f"Observation: Ferramenta '{chosen_tool_name}' não encontrada."
            else:
                next_prompt = "Observation: Ação não encontrada no formato esperado."

            print(next_prompt)
            continue

        if "Answer" in result:
            break

available_tools = {
    "get_convergence_rule": get_convergence_rule
}

agent_loop(client=client, tools=available_tools, max_iterations=10, query="A Série Harmônica converge ou diverge? Explique.")