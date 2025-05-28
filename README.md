## Detecção de Notícias Falsas com Técnicas de Processamento de Linguagem Natural e Aprendizado de Máquina

Este repositório contém o código-fonte do Trabalho de Conclusão de Curso apresentado à Faculdade de Computação da Universidade Federal de Uberlândia como parte dos requisitos exigidos para a obtenção do título de Bacharel em Ciência da Computação. 

Para acessar o trabalho: https://repositorio.ufu.br/handle/123456789/45951

## 👨‍🏫 Autora

- **Nome:** Larissa Alves Trevilato  
- **E-mail:** larissa.trevilato@gmail.com / larissa.alves@ufu.br  
- **Instituição:** Faculdade de Computação – Universidade Federal de Uberlândia  
- **Orientadora:** Maria Adriana Vidigal de Lima

## 📄 Sobre o Projeto

O objetivo deste trabalho é aplicar técnicas de representação textual (BoW, TF-IDF, Word2Vec) combinadas com algoritmos de aprendizado de máquina (SVC, Regressão Logística, Naive Bayes, Random Forest) para a tarefa de detecção de notícias falsas em diferentes corpora. 

O estudo visa comparar o desempenho das abordagens utilizadas e oferecer uma base experimental que possa ser expandida em pesquisas futuras.

## 📊 Resultados

Os resultados são apresentados por meio de gráficos de radar e heatmaps, analisando as métricas de acurácia, precisão, revocação e F1-score para cada combinação de técnica de representação e classificador.

## 🧪 Bases Utilizadas

- **Fake.BR** - https://github.com/roneysco/Fake.br-Corpus
- **FakeRecogna** - https://github.com/Gabriel-Lino-Garcia/FakeRecogna
- **FakeTrueBR** - https://github.com/jpchav98/FakeTrue.Br
- **BoatosBR** - https://github.com/Felipe-Harrison/boatos-br-corpus

## 📚 Contribuição para Estudos Futuros

Este projeto é código aberto e está licenciado sob a MIT License. Ele pode ser usado, modificado e distribuído livremente. Espera-se que este trabalho possa servir como base para pesquisas futuras nas áreas de PLN, aprendizado de máquina e combate à desinformação.

Se você usar ou se inspirar neste trabalho, sinta-se à vontade para citar este repositório ou entrar em contato.

## 📌 Como citar este trabalho

Caso queira citar diretamente, segundo as normas da **ABNT**: 

```TREVILATO, Larissa Alves. Análise comparativa de modelos de representação de texto e métodos de aprendizado de máquina na classificação de notícias falsas em português. 2025. Trabalho de Conclusão de Curso (Graduação em Ciência da Computação) – Universidade Federal de Uberlândia, Uberlândia, 2025. Orientadora: Maria Adriana Vidigal de Lima. Disponível em: https://repositorio.ufu.br/handle/123456789/45951. Acesso em: [coloque a data de acesso aqui].```

Caso você esteja usando Latex:

```
@misc{Trevilato:2025,
  author       = {Larissa Alves Trevilato},
  title        = {Análise Comparativa de Modelos de Representação de Texto e Métodos de Aprendizado de Máquina na Classificação de Notícias Falsas em Português},
  year         = {2025},
  address      = {Uberlândia, Brasil},
  note         = {Trabalho de Conclusão de Curso (Graduação em Ciência da Computação) – Universidade Federal de Uberlândia. Orientadora: Maria Adriana Vidigal de Lima},
  url = {https://repositorio.ufu.br/handle/123456789/45951}
  urldate = {coloque a data de acesso aqui}
}
```

## 📬 Contato

Caso tenha dúvidas ou sugestões, entre em contato pelo e-mail: [larissa.trevilato@gmail.com] ou abra uma issue neste repositório.

## ⭐ Se este projeto for útil para você, não esqueça de deixar uma estrela!

## 📁 Estrutura do Repositório




## 🛠️ Ambiente de Desenvolvimento

Todo o desenvolvimento deste trabalho foi realizado em linguagem **Python (versão 3.11.12)**, utilizando o ambiente computacional **Google Colab**, que oferece infraestrutura de execução em nuvem.
A principal biblioteca utilizada para a aplicação dos métodos de aprendizado de máquina foi a **scikit-learn**, em sua versão **1.6.1**.

## 🚀 Como Executar Localmente

1. Clone o repositório:

```
git clone https://github.com/laritrevilato/fake_news_classification.git
cd fake_news_classification/src

```

2. Crie um ambiente virtual:

```
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

```

3. Instale as bibliotecas necessárias:

```
pip install -r requirements.txt

```

4. Execute os notebooks ou scripts disponíveis em src/.
```


```

