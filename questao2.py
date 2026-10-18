"""
Trabalho 1 - Introdução ao Processamento de Imagens (IPI - UnB)
Questão 2: Realce de Contraste (car.png, crowd.png, university.png)

Implementação em Python (NumPy, OpenCV, Matplotlib) adaptada do enunciado original:
  2.1 Power-Law (Correção Gamma): s = c * r^gamma (via normalização [0, 1] e via cv2.LUT),
      testando 2 valores com gamma < 1 e 2 valores com gamma > 1 para cada imagem.
  2.2 Equalização de Histograma: cv2.equalizeHist nas três imagens, comparação visual e
      gráfico dedicado com Histograma + curva CDF antes e depois da equalização.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 2.1 Transformação Power-Law (Correção Gamma)
# =============================================================================

def correcao_gamma(
    img: np.ndarray,
    gamma: float,
    c: float = 1.0,
    usar_lut: bool = True,
) -> np.ndarray:
    """
    Aplica a transformação Power-Law (correção gamma):
        s = c * (r ^ gamma)
    onde r é a intensidade de entrada normalizada no intervalo [0, 1] e s é a
    intensidade de saída reescalonada para [0, 255] (uint8).

    Suporta duas estratégias equivalentes:
      - Look-Up Table (`cv2.LUT`): pré-calcula os 256 níveis possíveis de uint8
        em uma tabela 1D e mapeia a imagem em tempo O(N) sem operações de ponto
        flutuante por pixel.
      - Normalização direta em ponto flutuante (`usar_lut=False`): divide a matriz
        por 255.0, aplica a potência `r ** gamma` e multiplica por 255.0.

    Parâmetros
    ----------
    img : np.ndarray
        Imagem de entrada em níveis de cinza (H, W) ou colorida (H, W, C), dtype uint8.
    gamma : float
        Expoente gamma (> 0).
        - gamma < 1: expande tons escuros (clareia imagens subexpostas).
        - gamma > 1: expande tons claros / comprime tons escuros (escurece a imagem).
    c : float, opcional
        Constante multiplicativa da lei de potência (padrão = 1.0).
    usar_lut : bool, opcional
        Se True (padrão), utiliza `cv2.LUT` para máxima eficiência em imagens uint8.
        Se False, realiza o cálculo direto na matriz normalizada em [0, 1].

    Retorno
    -------
    np.ndarray
        Imagem realçada (uint8) com mesmas dimensões da entrada.
    """
    if gamma <= 0:
        raise ValueError(f"O parâmetro 'gamma' deve ser positivo (recebido: {gamma}).")

    if usar_lut and img.dtype == np.uint8:
        # Constrói a Look-Up Table para todas as 256 intensidades possíveis [0..255]
        r_lut = np.arange(256, dtype=np.float64) / 255.0
        s_lut = np.clip(c * (r_lut ** gamma) * 255.0, 0.0, 255.0)
        tabela_lut = np.round(s_lut).astype(np.uint8)
        return cv2.LUT(img, tabela_lut)

    # Alternativa: normalização explícita da imagem no intervalo [0, 1]
    r_norm = img.astype(np.float64) / 255.0
    s_norm = c * np.power(r_norm, gamma)
    saida = np.clip(np.round(s_norm * 255.0), 0.0, 255.0)
    return saida.astype(np.uint8)


# =============================================================================
# 2.2 Equalização de Histograma e Cálculo do Histograma / CDF
# =============================================================================

def equalizar_histograma(img: np.ndarray) -> np.ndarray:
    """
    Realiza a equalização global de histograma utilizando `cv2.equalizeHist`.
    Caso a imagem possua 3 canais (colorida), converte para escala de cinza ou
    equaliza o canal de luminância.

    Parâmetros
    ----------
    img : np.ndarray
        Imagem em escala de cinza (H, W) uint8.

    Retorno
    -------
    np.ndarray
        Imagem equalizada (H, W) uint8.
    """
    if img.ndim == 3:
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(img_cinza)
    return cv2.equalizeHist(img)


def calcular_histograma_e_cdf(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula o histograma de 256 níveis de cinza [0..255] e a respectiva
    Função de Distribuição Acumulada (CDF) normalizada no intervalo [0, 1].

    Parâmetros
    ----------
    img : np.ndarray
        Imagem em níveis de cinza (uint8).

    Retorno
    -------
    hist : np.ndarray
        Vetor (256,) com a contagem de pixels em cada nível de intensidade.
    cdf_norm : np.ndarray
        Vetor (256,) com a soma acumulada normalizada (de 0.0 a 1.0).
    """
    # Calcula o histograma via cv2.calcHist (ou np.bincount)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    total_pixels = cdf[-1] if cdf[-1] > 0 else 1.0
    cdf_norm = cdf / float(total_pixels)
    return hist, cdf_norm


# =============================================================================
# Funções de Carregamento e Visualização (Matplotlib)
# =============================================================================

def carregar_imagens_questao2(
    img_dir: str | Path = "img",
    nomes_arquivos: Sequence[str] = ("car.png", "crowd.png", "university.png"),
) -> Dict[str, np.ndarray]:
    """
    Carrega as imagens em escala de cinza da Questão 2 (car.png, crowd.png, university.png).
    """
    pasta = Path(img_dir)
    imagens: Dict[str, np.ndarray] = {}

    for nome in nomes_arquivos:
        caminho = pasta / nome
        if not caminho.exists():
            # Tenta no diretório atual caso img_dir não contenha o arquivo
            caminho_alt = Path(nome)
            if caminho_alt.exists():
                caminho = caminho_alt
            else:
                raise FileNotFoundError(
                    f"Imagem '{nome}' não encontrada em '{pasta.resolve()}'."
                )

        img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível decodificar a imagem '{caminho}'.")
        imagens[nome] = img

    return imagens


def plotar_power_law(
    imagens: Dict[str, np.ndarray],
    gammas: Sequence[float] = (0.4, 0.6, 1.5, 2.5),
    output_dir: str | Path = "resultados",
    mostrar: bool = True,
) -> Dict[str, Dict[float, np.ndarray]]:
    """
    Aplica a correção gamma com 2 valores < 1 e 2 valores > 1 para cada uma das
    três imagens e plota uma grade comparativa no Matplotlib.
    """
    pasta_saida = Path(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    n_imgs = len(imagens)
    n_cols = 1 + len(gammas)  # Coluna 0: Original + 4 colunas para os valores de gamma

    fig, eixos = plt.subplots(n_imgs, n_cols, figsize=(3.4 * n_cols, 3.2 * n_imgs))
    if n_imgs == 1:
        eixos = np.expand_dims(eixos, axis=0)

    fig.suptitle(
        r"Questão 2.1: Realce Power-Law ($s = c \cdot r^\gamma$) — Comparação $\gamma < 1$ vs $\gamma > 1$",
        fontsize=14,
        fontweight="bold",
    )

    resultados_gamma: Dict[str, Dict[float, np.ndarray]] = {}

    for i, (nome, img_orig) in enumerate(imagens.items()):
        resultados_gamma[nome] = {}

        # Imagem original
        eixos[i, 0].imshow(img_orig, cmap="gray", vmin=0, vmax=255)
        eixos[i, 0].set_title(rf"{nome}" + "\n" + r"Original ($\gamma = 1.0$)", fontsize=10)
        eixos[i, 0].axis("off")

        # Imagens corrigidas para cada valor de gamma
        for j, g in enumerate(gammas, start=1):
            img_gamma = correcao_gamma(img_orig, gamma=g, c=1.0, usar_lut=True)
            resultados_gamma[nome][g] = img_gamma

            tipo_gamma = r"\gamma < 1" if g < 1.0 else r"\gamma > 1"
            eixos[i, j].imshow(img_gamma, cmap="gray", vmin=0, vmax=255)
            eixos[i, j].set_title(rf"$\gamma = {g}$ (${tipo_gamma}$)", fontsize=10)
            eixos[i, j].axis("off")

    plt.tight_layout()
    caminho_fig = pasta_saida / "questao2_power_law_gamma.png"
    fig.savefig(caminho_fig, dpi=150, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return resultados_gamma


def plotar_equalizacao_comparacao(
    imagens: Dict[str, np.ndarray],
    melhores_gammas: Dict[str, float] | None = None,
    output_dir: str | Path = "resultados",
    mostrar: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Aplica `cv2.equalizeHist` nas três imagens e plota a comparação visual:
    Original vs Melhor Gamma vs Equalização de Histograma.
    """
    pasta_saida = Path(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    if melhores_gammas is None:
        melhores_gammas = {
            "car.png": 2.0,
            "crowd.png": 0.4,
            "university.png": 0.4,
        }

    n_imgs = len(imagens)
    fig, eixos = plt.subplots(n_imgs, 3, figsize=(13, 3.5 * n_imgs))
    if n_imgs == 1:
        eixos = np.expand_dims(eixos, axis=0)

    fig.suptitle(
        "Questão 2.2 e 2.3: Comparação Visual — Original vs Melhor Gamma vs Equalização de Histograma",
        fontsize=13,
        fontweight="bold",
    )

    resultados_eq: Dict[str, np.ndarray] = {}

    for i, (nome, img_orig) in enumerate(imagens.items()):
        img_eq = equalizar_histograma(img_orig)
        resultados_eq[nome] = img_eq

        melhor_g = melhores_gammas.get(nome, 0.5)
        img_melhor_gamma = correcao_gamma(img_orig, gamma=melhor_g, usar_lut=True)

        # Coluna 0: Original
        eixos[i, 0].imshow(img_orig, cmap="gray", vmin=0, vmax=255)
        eixos[i, 0].set_title(f"{nome} — Original", fontsize=11)
        eixos[i, 0].axis("off")

        # Coluna 1: Melhor Power-Law (Gamma)
        eixos[i, 1].imshow(img_melhor_gamma, cmap="gray", vmin=0, vmax=255)
        eixos[i, 1].set_title(rf"{nome} — Melhor Gamma ($\gamma = {melhor_g}$)", fontsize=11)
        eixos[i, 1].axis("off")

        # Coluna 2: Equalização de Histograma (cv2.equalizeHist)
        eixos[i, 2].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
        eixos[i, 2].set_title(f"{nome} — Equalização de Histograma", fontsize=11)
        eixos[i, 2].axis("off")

    plt.tight_layout()
    caminho_fig = pasta_saida / "questao2_equalizacao_comparacao.png"
    fig.savefig(caminho_fig, dpi=150, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return resultados_eq


def plotar_histograma_e_cdf(
    img_original: np.ndarray,
    img_equalizada: np.ndarray,
    nome_imagem: str = "car.png",
    output_dir: str | Path = "resultados",
    mostrar: bool = True,
) -> None:
    """
    Gera um gráfico dedicado mostrando a imagem, o Histograma e a curva CDF
    (Função de Distribuição Acumulada) antes e depois da equalização de histograma.
    """
    pasta_saida = Path(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    hist_antes, cdf_antes = calcular_histograma_e_cdf(img_original)
    hist_depois, cdf_depois = calcular_histograma_e_cdf(img_equalizada)
    niveis = np.arange(256)

    fig, eixos = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"Questão 2.2: Histograma e Curva CDF Antes e Depois da Equalização ({nome_imagem})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Imagem Original
    eixos[0, 0].imshow(img_original, cmap="gray", vmin=0, vmax=255)
    eixos[0, 0].set_title(f"Imagem Original ({nome_imagem})")
    eixos[0, 0].axis("off")

    # 2. Histograma + CDF ANTES da Equalização
    ax_hist_antes = eixos[0, 1]
    ax_cdf_antes = ax_hist_antes.twinx()

    ax_hist_antes.bar(
        niveis,
        hist_antes,
        width=1.0,
        color="steelblue",
        alpha=0.65,
        label="Histograma (Contagem)",
    )
    ax_cdf_antes.plot(
        niveis,
        cdf_antes,
        color="crimson",
        linewidth=2.2,
        label="CDF Acumulada Normalizada",
    )

    ax_hist_antes.set_title("Antes da Equalização: Histograma e CDF")
    ax_hist_antes.set_xlabel("Nível de Intensidade (0 a 255)")
    ax_hist_antes.set_ylabel("Número de Pixels", color="steelblue")
    ax_cdf_antes.set_ylabel("CDF Normalizada [0, 1]", color="crimson")
    ax_hist_antes.set_xlim(0, 255)
    ax_cdf_antes.set_ylim(0, 1.05)
    ax_hist_antes.grid(True, linestyle="--", alpha=0.3)

    # Combina as legendas dos dois eixos Y
    linhas1, rotulos1 = ax_hist_antes.get_legend_handles_labels()
    linhas2, rotulos2 = ax_cdf_antes.get_legend_handles_labels()
    ax_hist_antes.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc="upper left")

    # 3. Imagem Equalizada
    eixos[1, 0].imshow(img_equalizada, cmap="gray", vmin=0, vmax=255)
    eixos[1, 0].set_title(f"Imagem Equalizada via cv2.equalizeHist ({nome_imagem})")
    eixos[1, 0].axis("off")

    # 4. Histograma + CDF DEPOIS da Equalização
    ax_hist_depois = eixos[1, 1]
    ax_cdf_depois = ax_hist_depois.twinx()

    ax_hist_depois.bar(
        niveis,
        hist_depois,
        width=1.0,
        color="seagreen",
        alpha=0.65,
        label="Histograma Equalizado",
    )
    ax_cdf_depois.plot(
        niveis,
        cdf_depois,
        color="darkorange",
        linewidth=2.2,
        label="CDF Equalizada (Linearizada)",
    )

    ax_hist_depois.set_title("Depois da Equalização: Histograma e CDF")
    ax_hist_depois.set_xlabel("Nível de Intensidade (0 a 255)")
    ax_hist_depois.set_ylabel("Número de Pixels", color="seagreen")
    ax_cdf_depois.set_ylabel("CDF Normalizada [0, 1]", color="darkorange")
    ax_hist_depois.set_xlim(0, 255)
    ax_cdf_depois.set_ylim(0, 1.05)
    ax_hist_depois.grid(True, linestyle="--", alpha=0.3)

    linhas3, rotulos3 = ax_hist_depois.get_legend_handles_labels()
    linhas4, rotulos4 = ax_cdf_depois.get_legend_handles_labels()
    ax_hist_depois.legend(linhas3 + linhas4, rotulos3 + rotulos4, loc="upper left")

    plt.tight_layout()
    nome_limpo = Path(nome_imagem).stem
    caminho_fig = pasta_saida / f"questao2_histograma_cdf_{nome_limpo}.png"
    fig.savefig(caminho_fig, dpi=150, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def executar_questao2(
    img_dir: str | Path = "img",
    output_dir: str | Path = "resultados",
    gammas: Sequence[float] = (0.4, 0.6, 1.5, 2.5),
    imagem_cdf: str = "car.png",
    mostrar: bool = True,
) -> None:
    """
    Executa todo o pipeline da Questão 2:
      - Carrega car.png, crowd.png e university.png.
      - Testa 2 valores de gamma < 1 (0.4 e 0.6) e 2 valores de gamma > 1 (1.5 e 2.5).
      - Equaliza o histograma das 3 imagens com cv2.equalizeHist.
      - Gera os gráficos comparativos e o gráfico dedicado de Histograma + CDF.
    """
    imagens = carregar_imagens_questao2(img_dir)

    print("[OK] Questão 2 - Estatísticas das imagens originais:")
    for nome, img in imagens.items():
        print(
            f"  {nome:15s}: shape={img.shape}, min={img.min():3d}, "
            f"max={img.max():3d}, média={img.mean():6.2f}, std={img.std():5.2f}"
        )

    # 2.1 Realce Power-Law (Gamma)
    plotar_power_law(imagens, gammas=gammas, output_dir=output_dir, mostrar=mostrar)

    # 2.2 Equalização de Histograma nas três imagens
    resultados_eq = plotar_equalizacao_comparacao(
        imagens, output_dir=output_dir, mostrar=mostrar
    )

    # 2.2 Gráfico dedicado com Histograma e CDF antes e depois para a imagem selecionada
    if imagem_cdf not in imagens:
        imagem_cdf = next(iter(imagens.keys()))

    plotar_histograma_e_cdf(
        img_original=imagens[imagem_cdf],
        img_equalizada=resultados_eq[imagem_cdf],
        nome_imagem=imagem_cdf,
        output_dir=output_dir,
        mostrar=mostrar,
    )

    print(f"[OK] Questão 2 concluída. Gráficos salvos em '{Path(output_dir).resolve()}'.\n")


if __name__ == "__main__":
    executar_questao2(mostrar=True)
