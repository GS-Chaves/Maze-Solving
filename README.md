# Maze Solving

## Metodologia de desenvolvimento
- O programa foi organizado em torno de um agente que explora o labirinto em tempo real, registrando conhecimento parcial e atualizando um mapa interno a cada movimento.
- A estrategia principal combina exploracao em profundidade com heuristicas baseadas nas leituras laterais e diagonais do sensor, ajudando o agente a evitar trajetos repetitivos sempre que houver alternativas seguras.
- Um sensor radial opcional calcula, em tempo real, a distribuicao global de comida e orienta o agente para direcoes com maior densidade de recompensas, balanceando as outras heuristicas.
- As decisoes do agente levam em conta pontuacoes para comida, celulas desconhecidas e aproximacao da saida, alem de penalizar revisitas frequentes e caminhos previamente marcados como pouco promissores.

## Algoritmos empregados
- **Exploracao (DFS heuristico)**: o agente avanca com uma busca em profundidade, priorizando vizinhos a partir das heuristicas descritas e mantendo um historico de visitas para diminuir ciclos.
- **Planejamento de rotas (BFS)**: quando toda a comida possivel foi coletada, uma busca em largura garante um caminho conhecido ate a saida.
- **Sensoriamento lateral e diagonal**: as informacoes perimetrais sao usadas para ajustar prioridades, premiar corredores livres e evitar passos repetitivos ao reconhecer celulas ja observadas pelos lados.
- **Sensor radial de comida (opcional)**: agrega toda a comida restante em oito setores (N, NE, L, SE, S, SO, O, NO), ranqueando direcoes a partir da quantidade e proximidade dos itens para acelerar a coleta.

## Geração dos mapas
- A classe `MazeMap` constroi labirintos usando um algoritmo de carvamento em profundidade sobre uma malha reticulada, garantindo um caminho principal conectando entrada e saida.
- Em seguida sao adicionados loops e ilhas controlados por parametros de densidade, o que aumenta a complexidade sem comprometer a solubilidade.
- A distribuicao dos itens de comida considera tanto o caminho principal quanto bifurcacoes secundarias, evitando concentracao excessiva em uma unica area.
- Os mapas podem ser salvos em arquivo texto (`maze.txt` por padrao) e reutilizados em execucoes futuras.

## Requisitos
- Python 3.9 ou superior.
- Dependencias: `opencv-python` e `numpy`. Instale com `pip install opencv-python numpy`.

## Ambiente virtual (.venv)
- Crie o ambiente uma vez com `python3 -m venv .venv` (na pasta do projeto).
- Ative no Linux/macOS com `source .venv/bin/activate`; no Windows PowerShell use `.venv\Scripts\Activate.ps1`.
- Com o ambiente ativo, instale as dependencias `pip install opencv-python numpy` (ou `pip install -r requirements.txt` se existir).
- Sempre que for executar o programa, ative o `.venv` antes; para sair, utilize `deactivate`.

## Execução
1. Certifique-se de ter um arquivo `maze.txt` com o labirinto desejado na raiz do projeto.
2. Execute:
   ```
   python main.py
   ```
   Se nenhum parametro for informado e `maze.txt` existir, o programa utilizara esse mapa sem gerar um novo.
3. Ao final da execucao, um video MP4 sera criado com o percurso (`maze_run_medium.mp4`, por exemplo) e as informacoes de pontuacao serao exibidas no terminal.

Caso `maze.txt` nao exista, um novo labirinto sera gerado automaticamente de acordo com o tamanho padrao ou com os parametros fornecidos.

## Parametros de inicializacao
- `--maze-size {small|medium|large|extralarge}`: define o tamanho do labirinto a ser gerado. Dispara uma nova geracao mesmo se ja existir um arquivo.
- `--maze-file CAMINHO`: usa ou gera o labirinto em um arquivo especifico em vez do arquivo padrao `maze.txt`.
- `--video-file CAMINHO`: altera o caminho/arquivo MP4 de saida.
- `--regenerate`: força a geracao de um novo labirinto, sobrescrevendo o arquivo alvo.
- `--dump-frames`: salva cada frame como PNG em `frames_debug/` para inspecionar o percurso quadro a quadro.
- `--food-sensor` / `--no-food-sensor`: ativa ou desativa o sensor radial de comida que prioriza direcoes com maior concentracao de alimento.
- `--show-log` / `--no-log`: exibe (padrao) ou oculta o log detalhado dos movimentos do agente no terminal.

Variaveis de ambiente opcionais:
- `MAZE_SIZE`: define o tamanho do labirinto gerado quando `--maze-size` nao e informado (por exemplo, `MAZE_SIZE=large`).
- `MAZE_DUMP_FRAMES`: ativa `--dump-frames` quando ajustada para `1`, `true` ou `yes`.
- `MAZE_FOOD_SENSOR`: ativa o sensor radial de comida quando ajustada para `1`, `true`, `yes` ou `on`.

## Sensor radial de comida
- Quando ativado, o ambiente fornece um sensor 3x3 composto por oito setores que cobrem toda a matriz ate as bordas. Cada celula do sensor representa o total de alimentos naquela direcao (cardinal ou diagonal) e ignora paredes.
- As contagens sao alimentadas diretamente pelas posicoes reais das comidas restantes, incluindo distancias minimas em cada setor, permitindo ao agente pesar a densidade e a proximidade dos itens na heuristica de movimento.
- A heuristica adicional aumenta a prioridade de vizinhos alinhados com setores ricos em comida e reduz a dispersao desnecessaria, acelerando a coleta antes do caminho final para a saida.

## Saida e visualizacao
- O agente gera um video MP4 suave (`mp4v`) com grade redesenhada, destaques para entrada e saida, alimentos exibidos como marcadores circulares e um avatar com contorno e orientacao visivel.
- Opcionalmente, os frames individuais podem ser inspecionados na pasta `frames_debug/`.
- O terminal apresenta resumo da pontuacao, mapa interno aprendido e, se habilitado, o log completo de movimentos.
