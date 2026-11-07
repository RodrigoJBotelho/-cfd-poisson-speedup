# CFD Poisson Speedup

Simulação do **sistema linear da pressão** (equação de Poisson 2D), uma das etapas mais
custosas em CFD, com medição de **tempo de execução** e **speedup com múltiplas threads**
(C + OpenMP, sem dependências além do compilador).

## Contexto

Em simulações de escoamento incompressível pelo método de projeção, cada passo de tempo exige
resolver uma equação de Poisson para a pressão, o que gera um sistema linear grande e esparso.
Em muitos códigos essa etapa costuma concentrar boa parte do tempo de execução, por isso é um
bom alvo para paralelização e para estudar escalabilidade.

Este projeto isola essa etapa: resolve

```
∇²p = f      no quadrado unitário, com p = 0 no contorno
```

por diferenças finitas de 5 pontos. O termo-fonte vem de uma **solução manufaturada**,
`p(x,y) = sin(πx)·sin(πy)`, então é possível validar o resultado contra a solução exata.

## O que está implementado

- **Jacobi**: atualização de todos os pontos a partir da iteração anterior.
- **Red-Black SOR**: atualiza os pontos "vermelhos" e depois os "pretos". Pontos da mesma cor
  não dependem entre si, então não há condição de corrida. Usa o ω ótimo `2 / (1 + sin(πh))`.
- **Paralelismo com OpenMP**: divisão da malha por linhas, redução paralela para o resíduo e
  inicialização paralela da memória ("first touch").
- **Métricas**: tempo (melhor de R execuções), speedup `S = T(1)/T(p)`, eficiência `S/p`,
  métrica de **Karp-Flatt** (fração serial efetiva) e throughput em milhões de pontos por segundo.
- **Verificação de corretude** (`make test`): convergência, erro contra a solução exata,
  ordem de convergência 2 (o erro cai ~4x ao dobrar a resolução) e resultado idêntico com
  1 e 4 threads.

## Como rodar sem instalar nada (GitHub Actions)

1. Publique este repositório no GitHub.
2. Abra a aba **Actions** → workflow **benchmark** → **Run workflow** (você pode escolher
   tamanho da grade, número de iterações e método).
3. Ao terminar, abra a execução: a tabela de speedup aparece no **resumo** da página, e o
   CSV fica disponível em **Artifacts**.

O workflow também roda automaticamente a cada `push`, compilando, verificando e medindo.

## Como rodar localmente (Linux, ou macOS com gcc e OpenMP)

```bash
make            # compila
make test       # verificação de corretude
./poisson       # benchmark com 1024 x 1024, threads automáticas (1, 2, 4, ... núcleos)

./poisson --n 2048 --iter 300 --threads 1,2,4,8 --reps 5
./poisson --metodo rbsor --tol 1e-8 --iter 20000 --threads 1
./poisson --help
```

Por padrão o benchmark executa um **número fixo de iterações**, para que todas as contagens de
threads façam exatamente o mesmo trabalho; nesse modo a solução não precisa ter convergido.
Use `--tol` para iterar até convergir.

## Como interpretar os resultados

- Jacobi e Red-Black SOR fazem pouca conta por ponto lido da memória, então são limitados
  por **largura de banda de memória**. É normal o speedup **saturar** antes de usar todos os núcleos.
- Grades pequenas cabem no cache e escalam mal por causa do custo de sincronização entre threads.
- Núcleos lógicos (hyper-threading) não equivalem a núcleos físicos, e máquinas compartilhadas
  (como os runners do GitHub Actions) têm poucos núcleos e ruído de medição.
- Karp-Flatt estável com `p` indica que a fração serial explica a perda de escala; se ele
  cresce com `p`, o custo extra vem de sobrecarga (sincronização, memória).

## Resultados

Cole aqui a tabela gerada pelo workflow, indicando a data e o ambiente (por exemplo, runner
`ubuntu-latest`).

## Limitações

- Problema 2D, contorno de Dirichlet homogêneo e malha uniforme.
- Sem multigrid, gradiente conjugado, MPI ou GPU.
- Mede memória compartilhada (threads) em uma única máquina.

## Roteiro

- [ ] Gradiente Conjugado com pré-condicionador
- [ ] Multigrid geométrico (V-cycle)
- [ ] Versão 3D
- [ ] Gráfico de speedup a partir do CSV
- [ ] Comparação de métodos pelo tempo até convergir

## Licença

MIT. Veja [LICENSE](LICENSE).
