/*
 * poisson.c - Sistema linear da pressão (equação de Poisson 2D) com OpenMP.
 *
 * Resolve  laplaciano(p) = f  no quadrado unitário (contorno p = 0) com
 * diferenças finitas de 5 pontos, usando Jacobi ou Red-Black SOR, e mede
 * tempo de execução, speedup, eficiência e a métrica de Karp-Flatt para
 * diferentes números de threads.
 *
 * A fonte f vem de uma solução manufaturada, p(x,y) = sin(pi x) sin(pi y),
 * o que permite validar o resultado contra a solução exata.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846
#define MAX_THREADS_LISTA 64

typedef enum { JACOBI, RBSOR } Metodo;

typedef struct {
    int n;        /* pontos interiores por direção            */
    int ld;       /* n + 2 (inclui os pontos de contorno)     */
    double h;     /* espaçamento da malha = 1 / (n + 1)       */
    double *p;    /* solução atual                            */
    double *q;    /* buffer auxiliar (Jacobi)                 */
    double *f;    /* termo-fonte                              */
    double fnorm; /* norma de f, para o resíduo relativo      */
} Grade;

typedef struct {
    int iter;
    double tempo;
    double residuo;
    double erro;
} Resumo;

static const char *nome_metodo(Metodo m) { return m == JACOBI ? "jacobi" : "rbsor"; }

/* ------------------------------------------------------------------ */
/* Grade                                                               */
/* ------------------------------------------------------------------ */

static Grade grade_criar(int n) {
    Grade g;
    g.n = n;
    g.ld = n + 2;
    g.h = 1.0 / (n + 1);
    const size_t ld = (size_t)g.ld;
    const size_t tam = ld * ld;
    g.p = malloc(tam * sizeof(double));
    g.q = malloc(tam * sizeof(double));
    g.f = malloc(tam * sizeof(double));
    if (!g.p || !g.q || !g.f) {
        fprintf(stderr, "Erro: memória insuficiente para n = %d\n", n);
        exit(2);
    }

    double *p = g.p, *q = g.q, *f = g.f;
    const double h = g.h;
    const int nn = n;

    /* "first touch" paralelo: cada thread aloca as páginas das suas linhas. */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nn + 2; i++) {
        for (size_t j = 0; j < ld; j++) {
            p[(size_t)i * ld + j] = 0.0;
            q[(size_t)i * ld + j] = 0.0;
            f[(size_t)i * ld + j] = 0.0;
        }
    }

    double soma = 0.0;
#pragma omp parallel for reduction(+ : soma) schedule(static)
    for (int i = 1; i <= nn; i++) {
        for (int j = 1; j <= nn; j++) {
            const double v = -2.0 * PI * PI * sin(PI * i * h) * sin(PI * j * h);
            f[(size_t)i * ld + (size_t)j] = v;
            soma += v * v;
        }
    }
    g.fnorm = sqrt(soma);
    return g;
}

static void grade_liberar(Grade *g) {
    free(g->p);
    free(g->q);
    free(g->f);
}

static void grade_zerar(Grade *g) {
    const size_t ld = (size_t)g->ld;
    double *p = g->p, *q = g->q;
    const int nn = g->n;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nn + 2; i++) {
        for (size_t j = 0; j < ld; j++) {
            p[(size_t)i * ld + j] = 0.0;
            q[(size_t)i * ld + j] = 0.0;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Métodos iterativos                                                  */
/* ------------------------------------------------------------------ */

static void jacobi_passo(Grade *g) {
    const int n = g->n;
    const size_t ld = (size_t)g->ld;
    const double h2 = g->h * g->h;
    const double *restrict p = g->p;
    const double *restrict f = g->f;
    double *restrict q = g->q;

#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n; i++) {
        const size_t r = (size_t)i * ld;
        for (size_t j = 1; j <= (size_t)n; j++) {
            q[r + j] = 0.25 * (p[r + j - 1] + p[r + j + 1] + p[r - ld + j] + p[r + ld + j] -
                               h2 * f[r + j]);
        }
    }
    double *t = g->p;
    g->p = g->q;
    g->q = t;
}

/* Red-Black SOR: pontos da mesma cor não dependem entre si, então cada
 * cor pode ser atualizada em paralelo sem condição de corrida. */
static void rbsor_passo(Grade *g, double omega) {
    const int n = g->n;
    const size_t ld = (size_t)g->ld;
    const double h2 = g->h * g->h;
    double *restrict p = g->p;
    const double *restrict f = g->f;

    for (int cor = 0; cor < 2; cor++) {
#pragma omp parallel for schedule(static)
        for (int i = 1; i <= n; i++) {
            const size_t r = (size_t)i * ld;
            const int j0 = (((i + 1) & 1) == cor) ? 1 : 2; /* garante (i + j) % 2 == cor */
            for (int j = j0; j <= n; j += 2) {
                const double gs = 0.25 * (p[r + j - 1] + p[r + j + 1] + p[r - ld + j] +
                                          p[r + ld + j] - h2 * f[r + j]);
                p[r + j] += omega * (gs - p[r + j]);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Métricas de qualidade                                               */
/* ------------------------------------------------------------------ */

static double residuo_relativo(const Grade *g) {
    const int n = g->n;
    const size_t ld = (size_t)g->ld;
    const double h2 = g->h * g->h;
    const double *p = g->p;
    const double *f = g->f;
    double soma = 0.0;

#pragma omp parallel for reduction(+ : soma) schedule(static)
    for (int i = 1; i <= n; i++) {
        const size_t r = (size_t)i * ld;
        for (size_t j = 1; j <= (size_t)n; j++) {
            const double lap =
                (p[r + j - 1] + p[r + j + 1] + p[r - ld + j] + p[r + ld + j] - 4.0 * p[r + j]) / h2;
            const double res = f[r + j] - lap;
            soma += res * res;
        }
    }
    return sqrt(soma) / g->fnorm;
}

static double erro_maximo(const Grade *g) {
    const int n = g->n;
    const size_t ld = (size_t)g->ld;
    const double h = g->h;
    const double *p = g->p;
    double emax = 0.0;

#pragma omp parallel for reduction(max : emax) schedule(static)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            const double exato = sin(PI * i * h) * sin(PI * j * h);
            const double e = fabs(p[(size_t)i * ld + (size_t)j] - exato);
            if (e > emax) emax = e;
        }
    }
    return emax;
}

/* ------------------------------------------------------------------ */
/* Solver                                                              */
/* ------------------------------------------------------------------ */

/* tol <= 0: executa exatamente max_iter iterações (carga de trabalho fixa,
 * ideal para comparar threads). tol > 0: para quando o resíduo relativo < tol.
 * O tempo medido cobre apenas as iterações (e, se tol > 0, as checagens). */
static Resumo resolver(Grade *g, Metodo m, int max_iter, double tol) {
    grade_zerar(g);
    const double omega = (m == RBSOR) ? 2.0 / (1.0 + sin(PI * g->h)) : 1.0;
    int it = 0;

    const double t0 = omp_get_wtime();
    while (it < max_iter) {
        if (m == JACOBI)
            jacobi_passo(g);
        else
            rbsor_passo(g, omega);
        it++;
        if (tol > 0.0 && it % 20 == 0 && residuo_relativo(g) < tol) break;
    }
    const double t1 = omp_get_wtime();

    Resumo r;
    r.iter = it;
    r.tempo = t1 - t0;
    r.residuo = residuo_relativo(g);
    r.erro = erro_maximo(g);
    return r;
}

/* ------------------------------------------------------------------ */
/* Benchmark de speedup                                                */
/* ------------------------------------------------------------------ */

static int lista_threads_auto(int *lista) {
    const int max = omp_get_max_threads();
    int c = 0;
    for (int t = 1; t <= max && c < MAX_THREADS_LISTA; t *= 2) lista[c++] = t;
    if (lista[c - 1] != max && c < MAX_THREADS_LISTA) lista[c++] = max;
    return c;
}

static int lista_threads_parse(const char *s, int *lista) {
    if (strcmp(s, "auto") == 0) return lista_threads_auto(lista);
    int c = 0;
    char *copia = malloc(strlen(s) + 1);
    if (!copia) exit(2);
    strcpy(copia, s);
    for (char *tok = strtok(copia, ","); tok && c < MAX_THREADS_LISTA; tok = strtok(NULL, ",")) {
        const int t = atoi(tok);
        if (t < 1) {
            fprintf(stderr, "Erro: número de threads inválido: '%s'\n", tok);
            free(copia);
            exit(2);
        }
        lista[c++] = t;
    }
    free(copia);
    if (c == 0) {
        fprintf(stderr, "Erro: lista de threads vazia.\n");
        exit(2);
    }
    return c;
}

static int benchmark(int n, int iter, int reps, double tol, Metodo m, const char *threads_s,
                     const char *csv_path, const char *md_path) {
    int lista[MAX_THREADS_LISTA + 1];
    int c = lista_threads_parse(threads_s, lista);

    /* O speedup precisa da referência com 1 thread. */
    if (lista[0] != 1) {
        memmove(lista + 1, lista, (size_t)c * sizeof(int));
        lista[0] = 1;
        c++;
    }

    Grade g = grade_criar(n);

    FILE *csv = csv_path ? fopen(csv_path, "w") : NULL;
    FILE *md = md_path ? fopen(md_path, "w") : NULL;
    if ((csv_path && !csv) || (md_path && !md)) {
        fprintf(stderr, "Erro: não foi possível criar os arquivos de saída.\n");
        return 2;
    }

    const char *modo = tol > 0.0 ? "até convergir (tol)" : "número fixo de iterações";
    printf("Poisson 2D | grade %d x %d | método %s | %s | melhor de %d execuções\n", n, n,
           nome_metodo(m), modo, reps);
    printf("Núcleos lógicos detectados: %d\n\n", omp_get_num_procs());
    printf("%8s %12s %10s %12s %12s %12s\n", "threads", "tempo (s)", "speedup", "eficiência",
           "Karp-Flatt", "Mpts/s");

    if (md) {
        fprintf(md, "### Poisson 2D: speedup com OpenMP\n\n");
        fprintf(md, "Grade %d x %d, método `%s`, %s, melhor de %d execuções. ", n, n,
                nome_metodo(m), modo, reps);
        fprintf(md, "Núcleos lógicos detectados: %d.\n\n", omp_get_num_procs());
        fprintf(md, "| threads | tempo (s) | speedup | eficiência | Karp-Flatt | Mpts/s |\n");
        fprintf(md, "|---:|---:|---:|---:|---:|---:|\n");
    }
    if (csv)
        fprintf(csv, "threads,tempo_s,speedup,eficiencia,karp_flatt,mpts_s,iteracoes,residuo,erro_max\n");

    double t_base = 0.0;
    for (int k = 0; k < c; k++) {
        const int t = lista[k];
        omp_set_num_threads(t);

        double melhor = 1e300;
        Resumo ultimo = {0, 0, 0, 0};
        for (int r = 0; r < reps; r++) {
            ultimo = resolver(&g, m, iter, tol);
            if (ultimo.tempo < melhor) melhor = ultimo.tempo;
        }
        if (t == 1) t_base = melhor;

        const double speedup = t_base / melhor;
        const double eficiencia = speedup / t;
        const double kf = (t > 1) ? (1.0 / speedup - 1.0 / t) / (1.0 - 1.0 / t) : 0.0;
        const double mpts = (double)n * n * ultimo.iter / melhor / 1e6;

        printf("%8d %12.4f %10.2f %11.1f%% %12.3f %12.1f\n", t, melhor, speedup,
               100.0 * eficiencia, kf, mpts);
        if (md)
            fprintf(md, "| %d | %.4f | %.2f | %.1f%% | %.3f | %.1f |\n", t, melhor, speedup,
                    100.0 * eficiencia, kf, mpts);
        if (csv)
            fprintf(csv, "%d,%.6f,%.4f,%.4f,%.4f,%.2f,%d,%.3e,%.3e\n", t, melhor, speedup,
                    eficiencia, kf, mpts, ultimo.iter, ultimo.residuo, ultimo.erro);
    }

    printf("\nspeedup = T(1) / T(p); eficiência = speedup / p; "
           "Karp-Flatt estima a fração serial efetiva (quanto menor, melhor).\n");
    if (md)
        fprintf(md, "\n`speedup = T(1)/T(p)`, `eficiência = speedup/p`; Karp-Flatt estima a fração "
                    "serial efetiva.\n");

    if (csv) fclose(csv);
    if (md) fclose(md);
    grade_liberar(&g);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Verificação de corretude                                            */
/* ------------------------------------------------------------------ */

static int checar(int ok, const char *descricao, double valor) {
    printf("  [%s] %s (%.3e)\n", ok ? "OK" : "FALHOU", descricao, valor);
    return ok ? 0 : 1;
}

static int verificar(void) {
    int falhas = 0;
    printf("Verificação de corretude\n");

    omp_set_num_threads(1);

    /* 1. RB-SOR converge e fica próximo da solução exata. */
    Grade g63 = grade_criar(63);
    Resumo r63 = resolver(&g63, RBSOR, 5000, 1e-12);
    falhas += checar(r63.residuo < 1e-11, "RB-SOR (n=63): resíduo relativo < 1e-11", r63.residuo);
    falhas += checar(r63.erro < 1e-3, "RB-SOR (n=63): erro máximo vs solução exata < 1e-3", r63.erro);

    /* 2. Convergência de segunda ordem: erro cai ~4x ao dobrar a resolução. */
    Grade g31 = grade_criar(31);
    Resumo r31 = resolver(&g31, RBSOR, 5000, 1e-12);
    const double razao = r31.erro / r63.erro;
    falhas += checar(razao > 3.5 && razao < 4.5, "ordem 2: erro(n=31) / erro(n=63) entre 3,5 e 4,5", razao);
    grade_liberar(&g31);

    /* 3. Jacobi também converge. */
    Grade gj = grade_criar(31);
    Resumo rj = resolver(&gj, JACOBI, 20000, 1e-9);
    falhas += checar(rj.residuo < 1e-9, "Jacobi (n=31): resíduo relativo < 1e-9", rj.residuo);
    grade_liberar(&gj);
    grade_liberar(&g63);

    /* 4. Resultado idêntico com 1 e 4 threads (atualizações ponto a ponto). */
    const Metodo metodos[2] = {JACOBI, RBSOR};
    for (int k = 0; k < 2; k++) {
        Grade g = grade_criar(127);
        const size_t tam = (size_t)g.ld * (size_t)g.ld;
        double *ref = malloc(tam * sizeof(double));
        if (!ref) exit(2);

        omp_set_num_threads(1);
        resolver(&g, metodos[k], 100, 0.0);
        memcpy(ref, g.p, tam * sizeof(double));

        omp_set_num_threads(4);
        resolver(&g, metodos[k], 100, 0.0);
        double dmax = 0.0;
        for (size_t i = 0; i < tam; i++) {
            const double d = fabs(g.p[i] - ref[i]);
            if (d > dmax) dmax = d;
        }
        char desc[96];
        snprintf(desc, sizeof desc, "%s: diferença entre 1 e 4 threads == 0", nome_metodo(metodos[k]));
        falhas += checar(dmax == 0.0, desc, dmax);
        free(ref);
        grade_liberar(&g);
    }

    printf(falhas ? "\n%d verificação(ões) falhou(aram).\n" : "\nTodas as verificações passaram.\n",
           falhas);
    return falhas ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/* Linha de comando                                                    */
/* ------------------------------------------------------------------ */

static void uso(const char *prog) {
    printf("Uso: %s [opções]\n\n"
           "  --n N            pontos interiores por direção (padrão 1024)\n"
           "  --iter K         iterações por execução (padrão 500)\n"
           "  --tol T          se > 0, itera até o resíduo relativo < T (padrão 0 = iterações fixas)\n"
           "  --metodo M       jacobi | rbsor (padrão jacobi)\n"
           "  --threads L      lista separada por vírgulas, ex.: 1,2,4,8, ou 'auto' (padrão)\n"
           "  --reps R         execuções por contagem de threads; usa a melhor (padrão 3)\n"
           "  --csv ARQ        grava os resultados em CSV\n"
           "  --markdown ARQ   grava a tabela em Markdown\n"
           "  --verificar      roda os testes de corretude e sai\n"
           "  --help           mostra esta ajuda\n",
           prog);
}

int main(int argc, char **argv) {
    int n = 1024, iter = 500, reps = 3, fazer_verificar = 0;
    double tol = 0.0;
    Metodo m = JACOBI;
    const char *threads_s = "auto", *csv = NULL, *md = NULL;

    for (int a = 1; a < argc; a++) {
        const char *arg = argv[a];
        const int tem_valor = a + 1 < argc;
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            uso(argv[0]);
            return 0;
        } else if (!strcmp(arg, "--verificar")) {
            fazer_verificar = 1;
        } else if (!strcmp(arg, "--n") && tem_valor) {
            n = atoi(argv[++a]);
        } else if (!strcmp(arg, "--iter") && tem_valor) {
            iter = atoi(argv[++a]);
        } else if (!strcmp(arg, "--reps") && tem_valor) {
            reps = atoi(argv[++a]);
        } else if (!strcmp(arg, "--tol") && tem_valor) {
            tol = atof(argv[++a]);
        } else if (!strcmp(arg, "--threads") && tem_valor) {
            threads_s = argv[++a];
        } else if (!strcmp(arg, "--csv") && tem_valor) {
            csv = argv[++a];
        } else if (!strcmp(arg, "--markdown") && tem_valor) {
            md = argv[++a];
        } else if (!strcmp(arg, "--metodo") && tem_valor) {
            const char *v = argv[++a];
            if (!strcmp(v, "jacobi")) m = JACOBI;
            else if (!strcmp(v, "rbsor")) m = RBSOR;
            else {
                fprintf(stderr, "Erro: método desconhecido '%s' (use jacobi ou rbsor)\n", v);
                return 2;
            }
        } else {
            fprintf(stderr, "Argumento inválido: %s\n\n", arg);
            uso(argv[0]);
            return 2;
        }
    }

    if (fazer_verificar) return verificar();

    if (n < 3 || iter < 1 || reps < 1) {
        fprintf(stderr, "Erro: use n >= 3, iter >= 1 e reps >= 1.\n");
        return 2;
    }
    return benchmark(n, iter, reps, tol, m, threads_s, csv, md);
}
