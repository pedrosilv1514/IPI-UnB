"""
Trabalho 1 - Introdução ao Processamento de Imagens (IPI - UnB)
Questão 1: Escalonamento e Super-Resolução (FRUIT1.jpg e FRUIT2.jpg)

Implementação em Python (NumPy, OpenCV, Matplotlib) adaptada do enunciado original em MATLAB:
  1.1 tam2(img, fator): Escalonamento por repetição de vizinho (Nearest Neighbor) via indexação NumPy.
  1.2 tamm(img): Escalonamento 2x por interpolação bilinear sequencial (média horizontal seguida de vertical).
  1.3 superres(img1, img2): Super-resolução 2x combinando duas imagens deslocadas em grade intercalada.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1.1 Função tam2(img, fator) e função base de duplicação
# =============================================================================

def dobrar_tamanho_vizinho(img: np.ndarray) -> np.ndarray:
    """
    Função base que dobra a dimensão espacial (2x) de uma imagem (2D ou 3D RGB)
    repetindo a linha e a coluna anterior exclusivamente via indexação de fatias
    (slicing) do NumPy, sem utilizar cv2.resize ou np.repeat.

    Exemplo (2x2 -> 4x4):
        [[10, 12],       [[10, 10, 12, 12],
         [ 5,  7]]  -->   [10, 10, 12, 12],
                          [ 5,  5,  7,  7],
                          [ 5,  5,  7,  7]]

    Parâmetros
    ----------
    img : np.ndarray
        Imagem de entrada com shape (H, W) ou (H, W, C).

    Retorno
    -------
    np.ndarray
        Imagem escalonada com shape (2*H, 2*W) ou (2*H, 2*W, C).
    """
    altura, largura = img.shape[:2]
    canais = img.shape[2:]  # () para grayscale ou (C,) para RGB

    saida = np.zeros((2 * altura, 2 * largura, *canais), dtype=img.dtype)

    # Linhas pares recebem os pixels originais nas colunas pares
    # e repetem a coluna anterior nas colunas ímpares:
    saida[0::2, 0::2] = img
    saida[0::2, 1::2] = img

    # Linhas ímpares repetem integralmente a linha par imediatamente anterior:
    saida[1::2, :] = saida[0::2, :]

    return saida


def tam2(img: np.ndarray, fator: int) -> np.ndarray:
    """
    Aumenta o número de linhas e colunas da imagem pelo 'fator' especificado
    chamando recursivamente/iterativamente a função base `dobrar_tamanho_vizinho`.

    Parâmetros
    ----------
    img : np.ndarray
        Imagem colorida em RGB (H, W, 3) ou matriz 2D (H, W).
    fator : int
        Número par maior ou igual a 2 (ex.: 2, 4, 8).
        Para potências de 2 (como 2 e 8 exigidos no enunciado), aplica a
        duplicação exata log2(fator) vezes.

    Retorno
    -------
    np.ndarray
        Imagem redimensionada por repetição de linha/coluna anterior.
    """
    if not isinstance(fator, (int, np.integer)) or fator < 2 or fator % 2 != 0:
        raise ValueError(
            f"O parâmetro 'fator' deve ser um inteiro par maior ou igual a 2 (recebido: {fator})."
        )

    # Caso base da recursão: fator == 2
    if fator == 2:
        return dobrar_tamanho_vizinho(img)

    # Se o fator for potência de 2 (ex.: 4, 8, 16), aplica recursivamente dobrar_tamanho_vizinho
    if (fator & (fator - 1)) == 0:
        img_dobrada = dobrar_tamanho_vizinho(img)
        return tam2(img_dobrada, fator // 2)

    # Caso geral para qualquer número par (ex.: 6) via indexação direta por divisão inteira
    altura, largura = img.shape[:2]
    indices_linhas = np.arange(altura * fator) // fator
    indices_colunas = np.arange(largura * fator) // fator
    return img[indices_linhas[:, None], indices_colunas[None, :]]


# =============================================================================
# 1.2 Função tamm(img) - Interpolação por Média de Vizinhos (2x)
# =============================================================================

def _arredondar_matlab(valores: np.ndarray) -> np.ndarray:
    """
    Reproduz o arredondamento 'round half away from zero' (0.5 arredonda para cima)
    padrão da aritmética inteira uint8 do MATLAB, garantindo que:
      (10 + 5) / 2 = 7.5 -> 8
      (11 + 6) / 2 = 8.5 -> 9
      (12 + 7) / 2 = 9.5 -> 10
    """
    return np.floor(valores + 0.5)


def tamm(img: np.ndarray) -> np.ndarray:
    """
    Aumenta a imagem em 2x preenchendo:
      1º) Os pixels intermediários das linhas NÃO vazias (linhas pares na indexação 0-based)
          pela média horizontal dos vizinhos laterais, repetindo a borda direita.
      2º) As linhas vazias (linhas ímpares na indexação 0-based) pela média vertical
          das linhas superior e inferior já preenchidas, repetindo a borda inferior.

    Reproduz exatamente a transformação do enunciado:
        [[10, 12],       [[10, 11, 12, 12],
         [ 5,  7]]  -->   [ 8,  9, 10, 10],
                          [ 5,  6,  7,  7],
                          [ 5,  6,  7,  7]]

    Parâmetros
    ----------
    img : np.ndarray
        Imagem de entrada (H, W) ou (H, W, C).

    Retorno
    -------
    np.ndarray
        Imagem ampliada em 2x (2*H, 2*W, ...) com mesmo dtype da entrada.
    """
    altura, largura = img.shape[:2]
    canais = img.shape[2:]

    # Utiliza float64 durante os cálculos para evitar overflow em somas de uint8
    saida = np.zeros((2 * altura, 2 * largura, *canais), dtype=np.float64)

    # Passo 0: Aloca os pixels originais nas linhas pares e colunas pares
    saida[0::2, 0::2] = img.astype(np.float64)

    # Passo 1 (Pixels em vermelho no enunciado):
    # Preenche as colunas ímpares nas linhas NÃO vazias (linhas pares: 0, 2, 4, ...)
    # - Colunas ímpares internas (1, 3, ..., 2*W - 3): média entre esquerda e direita
    saida[0::2, 1:-1:2] = _arredondar_matlab(
        (saida[0::2, 0:-2:2] + saida[0::2, 2::2]) / 2.0
    )
    # - Última coluna (borda direita, índice 2*W - 1): repete a coluna anterior (2*W - 2)
    saida[0::2, -1] = saida[0::2, -2]

    # Passo 2 (Pixels em azul no enunciado):
    # Preenche as linhas totalmente vazias (linhas ímpares: 1, 3, 5, ...)
    # - Linhas ímpares internas (1, 3, ..., 2*H - 3): média entre linha acima e linha abaixo
    saida[1:-1:2, :] = _arredondar_matlab(
        (saida[0:-2:2, :] + saida[2::2, :]) / 2.0
    )
    # - Última linha (borda inferior, índice 2*H - 1): repete a linha anterior (2*H - 2)
    saida[-1, :] = saida[-2, :]

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        saida = np.clip(saida, info.min, info.max)

    return saida.astype(img.dtype)


# =============================================================================
# 1.3 Função superres(img1, img2) - Super-Resolução em Grade 2x
# =============================================================================

def superres(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Combina duas imagens de mesma dimensão (img1 e img2) em uma grade 2x maior.

    Mapeamento de índices (MATLAB base-1 -> Python base-0):
      - No MATLAB (base-1): img1 ocupa posições ímpares (1,1), (1,3), (3,3)...
        No Python (base-0): equivale aos índices PARES -> saida[0::2, 0::2] = img1
      - No MATLAB (base-1): img2 ocupa posições pares (2,2), (2,4), (4,4)...
        No Python (base-0): equivale aos índices ÍMPARES -> saida[1::2, 1::2] = img2

    Os pixels restantes nas posições (0::2, 1::2) e (1::2, 0::2) formam o complemento
    do tabuleiro de xadrez (grade quincunx) e são interpolados pela média aritmética
    dos seus vizinhos ortogonais disponíveis (4 vizinhos no interior, 3 nas bordas
    e 2 nos cantos superior-direito e inferior-esquerdo).

    Parâmetros
    ----------
    img1 : np.ndarray
        Primeira imagem de entrada (H, W) ou (H, W, C).
    img2 : np.ndarray
        Segunda imagem de entrada com mesmas dimensões de img1.

    Retorno
    -------
    np.ndarray
        Imagem de super-resolução com dobro de linhas e colunas (2*H, 2*W, ...).
    """
    if img1.shape != img2.shape:
        # Compatibiliza a dimensão espacial de img2 com img1 caso tenham tamanhos distintos
        img2 = tam2_ajustar_dimensao(img2, img1.shape[0], img1.shape[1])

    altura, largura = img1.shape[:2]
    canais = img1.shape[2:]

    saida = np.zeros((2 * altura, 2 * largura, *canais), dtype=np.float64)

    # 1. Aloca img1 nos índices pares (0::2, 0::2) e img2 nos índices ímpares (1::2, 1::2)
    f_img1 = img1.astype(np.float64)
    f_img2 = img2.astype(np.float64)
    saida[0::2, 0::2] = f_img1
    saida[1::2, 1::2] = f_img2

    # 2. Interpola os pixels (0::2, 1::2) [linhas pares, colunas ímpares]
    # Vizinhos conhecidos:
    # - Esquerda (sempre existe): (0::2, 0::2) -> f_img1
    # - Direita (existe exceto na última coluna): (0::2, 2::2) -> f_img1[:, 1:]
    # - Abaixo (sempre existe): (1::2, 1::2) -> f_img2
    # - Acima (existe exceto na primeira linha): (1:-1:2, 1::2) -> f_img2[:-1, :]
    soma_par_impar = f_img1 + f_img2  # Vizinhos da esquerda e de baixo sempre existem
    cont_par_impar = np.full((altura, largura, *(1 for _ in canais)), 2.0, dtype=np.float64)

    # Adiciona vizinho da direita (para colunas 0 até W-2)
    soma_par_impar[:, :-1] += f_img1[:, 1:]
    cont_par_impar[:, :-1] += 1.0

    # Adiciona vizinho de cima (para linhas 1 até H-1)
    soma_par_impar[1:, :] += f_img2[:-1, :]
    cont_par_impar[1:, :] += 1.0

    saida[0::2, 1::2] = _arredondar_matlab(soma_par_impar / cont_par_impar)

    # 3. Interpola os pixels (1::2, 0::2) [linhas ímpares, colunas pares]
    # Vizinhos conhecidos:
    # - Acima (sempre existe): (0::2, 0::2) -> f_img1
    # - Abaixo (existe exceto na última linha): (2::2, 0::2) -> f_img1[1:, :]
    # - Direita (sempre existe): (1::2, 1::2) -> f_img2
    # - Esquerda (existe exceto na primeira coluna): (1::2, 1:-1:2) -> f_img2[:, :-1]
    soma_impar_par = f_img1 + f_img2  # Vizinhos de cima e da direita sempre existem
    cont_impar_par = np.full((altura, largura, *(1 for _ in canais)), 2.0, dtype=np.float64)

    # Adiciona vizinho de baixo (para linhas 0 até H-2)
    soma_impar_par[:-1, :] += f_img1[1:, :]
    cont_impar_par[:-1, :] += 1.0

    # Adiciona vizinho da esquerda (para colunas 1 até W-1)
    soma_impar_par[:, 1:] += f_img2[:, :-1]
    cont_impar_par[:, 1:] += 1.0

    saida[1::2, 0::2] = _arredondar_matlab(soma_impar_par / cont_impar_par)

    if np.issubdtype(img1.dtype, np.integer):
        info = np.iinfo(img1.dtype)
        saida = np.clip(saida, info.min, info.max)

    return saida.astype(img1.dtype)


# =============================================================================
# Validação Numérica com as Matrizes do Enunciado e Carregamento de Imagens
# =============================================================================

def validar_matrizes_enunciado() -> None:
    """
    Verifica se as funções tam2, tamm e superres reproduzem exatamente os
    exemplos numéricos descritos no PDF do Trabalho 1.
    """
    matriz_base = np.array([[10, 12], [5, 7]], dtype=np.uint8)

    esperado_tam2 = np.array(
        [
            [10, 10, 12, 12],
            [10, 10, 12, 12],
            [5, 5, 7, 7],
            [5, 5, 7, 7],
        ],
        dtype=np.uint8,
    )
    esperado_tamm = np.array(
        [
            [10, 11, 12, 12],
            [8, 9, 10, 10],
            [5, 6, 7, 7],
            [5, 6, 7, 7],
        ],
        dtype=np.uint8,
    )

    obtido_tam2 = tam2(matriz_base, 2)
    obtido_tamm = tamm(matriz_base)

    assert np.array_equal(obtido_tam2, esperado_tam2), (
        f"Erro em tam2:\nObtido:\n{obtido_tam2}\nEsperado:\n{esperado_tam2}"
    )
    assert np.array_equal(obtido_tamm, esperado_tamm), (
        f"Erro em tamm:\nObtido:\n{obtido_tamm}\nEsperado:\n{esperado_tamm}"
    )

    # Teste de mapeamento de índices de superres conforme exemplo do PDF
    img1_teste = np.array([[10, 12], [5, 7]], dtype=np.uint8)
    img2_teste = np.array([[11, 13], [6, 8]], dtype=np.uint8)
    obtido_sr = superres(img1_teste, img2_teste)
    assert np.array_equal(obtido_sr[0::2, 0::2], img1_teste), "Erro na alocação par de superres"
    assert np.array_equal(obtido_sr[1::2, 1::2], img2_teste), "Erro na alocação ímpar de superres"

    print("[OK] Questão 1 - Matrizes de validação do enunciado verificadas com sucesso:")
    print(f"  Original (2x2):\n{matriz_base}")
    print(f"  tam2(img, 2) (4x4):\n{obtido_tam2}")
    print(f"  tamm(img) (4x4):\n{obtido_tamm}")
    print(f"  superres(img1, img2) (4x4):\n{obtido_sr}\n")


def _gerar_par_fruit_sintetico(altura: int = 128, largura: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera um par de imagens RGB sintéticas de alta frequência (FRUIT1 e FRUIT2)
    amostradas a partir de uma cena de resolução dupla (2*H, 2*W), caso os arquivos
    FRUIT1.jpg e FRUIT2.jpg não estejam presentes na pasta de imagens.
    """
    h_hr, w_hr = 2 * altura, 2 * largura
    y, x = np.ogrid[:h_hr, :w_hr]

    # Fundo suave com gradiente
    img_hr = np.zeros((h_hr, w_hr, 3), dtype=np.uint8)
    img_hr[..., 0] = np.clip(40 + (x * 60) // w_hr, 0, 255)
    img_hr[..., 1] = np.clip(50 + (y * 60) // h_hr, 0, 255)
    img_hr[..., 2] = 60

    # "Maçã" vermelha com iluminação especular
    cy1, cx1, r1 = int(h_hr * 0.52), int(w_hr * 0.42), int(min(h_hr, w_hr) * 0.28)
    dist1 = np.sqrt((y - cy1) ** 2 + (x - cx1) ** 2)
    mascara_maca = dist1 <= r1
    sombra_maca = np.clip(1.0 - (dist1 / (r1 + 1e-5)) ** 2, 0, 1)
    img_hr[mascara_maca, 0] = np.clip(180 + 75 * sombra_maca[mascara_maca], 0, 255).astype(np.uint8)
    img_hr[mascara_maca, 1] = np.clip(25 + 30 * sombra_maca[mascara_maca], 0, 255).astype(np.uint8)
    img_hr[mascara_maca, 2] = np.clip(20 + 25 * sombra_maca[mascara_maca], 0, 255).astype(np.uint8)

    # "Laranja" cítrica com textura senoidal fina (alta frequência para evidenciar aliasing/interpolação)
    cy2, cx2, r2 = int(h_hr * 0.60), int(w_hr * 0.68), int(min(h_hr, w_hr) * 0.22)
    dist2 = np.sqrt((y - cy2) ** 2 + (x - cx2) ** 2)
    mascara_laranja = dist2 <= r2
    textura = 12.0 * np.sin(0.6 * x) * np.cos(0.6 * y)
    sombra_laranja = np.clip(1.0 - 0.6 * (dist2 / (r2 + 1e-5)) ** 2, 0, 1)
    img_hr[mascara_laranja, 0] = np.clip(
        240 * sombra_laranja[mascara_laranja] + textura[mascara_laranja], 0, 255
    ).astype(np.uint8)
    img_hr[mascara_laranja, 1] = np.clip(
        140 * sombra_laranja[mascara_laranja] + textura[mascara_laranja], 0, 255
    ).astype(np.uint8)
    img_hr[mascara_laranja, 2] = np.clip(
        20 * sombra_laranja[mascara_laranja], 0, 255
    ).astype(np.uint8)

    # Amostragem sub-pixel: FRUIT1 nos índices pares e FRUIT2 nos índices ímpares
    fruit1 = img_hr[0::2, 0::2].copy()
    fruit2 = img_hr[1::2, 1::2].copy()
    return fruit1, fruit2


def tam2_ajustar_dimensao(img: np.ndarray, altura_alvo: int, largura_alvo: int) -> np.ndarray:
    """
    Compatibiliza a dimensão espacial de uma imagem para (altura_alvo, largura_alvo)
    via mapeamento proporcional de índices do NumPy.
    """
    h_orig, w_orig = img.shape[:2]
    idx_y = (np.arange(altura_alvo) * h_orig) // altura_alvo
    idx_x = (np.arange(largura_alvo) * w_orig) // largura_alvo
    return img[idx_y[:, None], idx_x[None, :]]


def carregar_imagens_fruit(img_dir: str | Path = "img") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega FRUIT1 (.jpg/.jpeg/.png) e FRUIT2 (.jpg/.jpeg/.png) no formato RGB.
    Busca de forma case-insensitive em `img_dir` e no diretório atual.
    """
    pasta = Path(img_dir)
    pasta.mkdir(parents=True, exist_ok=True)

    def _buscar_por_stem(stem_alvo: str) -> Path | None:
        extensoes_validas = {".jpg", ".jpeg", ".png", ".bmp"}
        for base in (pasta, Path(".")):
            if not base.exists():
                continue
            for arq in sorted(base.iterdir()):
                if (
                    arq.is_file()
                    and arq.stem.lower() == stem_alvo.lower()
                    and arq.suffix.lower() in extensoes_validas
                ):
                    return arq
        return None

    caminho_f1 = _buscar_por_stem("FRUIT1")
    caminho_f2 = _buscar_por_stem("FRUIT2")

    if caminho_f1 is not None and caminho_f2 is not None:
        bgr1 = cv2.imread(str(caminho_f1), cv2.IMREAD_COLOR)
        bgr2 = cv2.imread(str(caminho_f2), cv2.IMREAD_COLOR)
        if bgr1 is not None and bgr2 is not None:
            rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
            if rgb1.shape[:2] != rgb2.shape[:2]:
                rgb2 = tam2_ajustar_dimensao(rgb2, rgb1.shape[0], rgb1.shape[1])
            return rgb1, rgb2

    print(
        "[AVISO] 'FRUIT1' e/ou 'FRUIT2' não encontrados em 'img/'. "
        "Gerando par sintético de teste em 'img/FRUIT1.jpg' e 'img/FRUIT2.jpg'."
    )
    fruit1, fruit2 = _gerar_par_fruit_sintetico()
    cv2.imwrite(str(pasta / "FRUIT1.jpg"), cv2.cvtColor(fruit1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(pasta / "FRUIT2.jpg"), cv2.cvtColor(fruit2, cv2.COLOR_RGB2BGR))
    return fruit1, fruit2


def executar_questao1(
    img_dir: str | Path = "img",
    output_dir: str | Path = "resultados",
    mostrar: bool = True,
) -> dict[str, np.ndarray]:
    """
    Executa todos os itens da Questão 1 (1.1, 1.2 e 1.3), valida as matrizes do
    enunciado e gera as figuras comparativas de escalonamento e super-resolução.
    """
    validar_matrizes_enunciado()

    pasta_saida = Path(output_dir)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    fruit1, fruit2 = carregar_imagens_fruit(img_dir)

    # 1.1 Escalonamento por repetição (fator 2 e fator 8)
    fruit1_tam2_2x = tam2(fruit1, 2)
    fruit1_tam2_8x = tam2(fruit1, 8)

    # 1.2 Escalonamento 2x por média sequencial (tamm)
    fruit1_tamm_2x = tamm(fruit1)

    # 1.3 Super-resolução 2x combinando FRUIT1 e FRUIT2
    fruit_superres_2x = superres(fruit1, fruit2)

    print("Dimensões resultantes na Questão 1:")
    print(f"  FRUIT1 original      : {fruit1.shape}")
    print(f"  1.1 tam2(FRUIT1, 2)  : {fruit1_tam2_2x.shape}")
    print(f"  1.1 tam2(FRUIT1, 8)  : {fruit1_tam2_8x.shape}")
    print(f"  1.2 tamm(FRUIT1)     : {fruit1_tamm_2x.shape}")
    print(f"  1.3 superres(F1, F2) : {fruit_superres_2x.shape}\n")

    # Figura 1: Comparação geral dos métodos da Questão 1
    fig, eixos = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Questão 1: Escalonamento (tam2, tamm) e Super-Resolução (superres)",
        fontsize=14,
        fontweight="bold",
    )

    eixos[0, 0].imshow(fruit1, interpolation="nearest")
    eixos[0, 0].set_title(f"FRUIT1 Original {fruit1.shape[:2]}")
    eixos[0, 0].axis("off")

    eixos[0, 1].imshow(fruit1_tam2_2x, interpolation="nearest")
    eixos[0, 1].set_title(f"1.1 tam2(FRUIT1, 2) {fruit1_tam2_2x.shape[:2]}")
    eixos[0, 1].axis("off")

    eixos[0, 2].imshow(fruit1_tam2_8x, interpolation="nearest")
    eixos[0, 2].set_title(f"1.1 tam2(FRUIT1, 8) {fruit1_tam2_8x.shape[:2]}")
    eixos[0, 2].axis("off")

    eixos[1, 0].imshow(fruit2, interpolation="nearest")
    eixos[1, 0].set_title(f"FRUIT2 (Ajustada) {fruit2.shape[:2]}")
    eixos[1, 0].axis("off")

    eixos[1, 1].imshow(fruit1_tamm_2x, interpolation="nearest")
    eixos[1, 1].set_title(f"1.2 tamm(FRUIT1) 2x {fruit1_tamm_2x.shape[:2]}")
    eixos[1, 1].axis("off")

    eixos[1, 2].imshow(fruit_superres_2x, interpolation="nearest")
    eixos[1, 2].set_title(f"1.3 superres(FRUIT1, FRUIT2) 2x {fruit_superres_2x.shape[:2]}")
    eixos[1, 2].axis("off")

    plt.tight_layout()
    caminho_fig1 = pasta_saida / "questao1_comparacao.png"
    fig.savefig(caminho_fig1, dpi=150, bbox_inches="tight")

    # Figura 2: Zoom em região de borda/caule na mesma escala de pixels de saída
    # para evidenciar blocos 2x2 (tam2 2x), blocos 8x8 (tam2 8x), suavização (tamm) e fusão (superres)
    h2, w2 = fruit1_tam2_2x.shape[:2]
    r0, r1 = int(h2 * 0.14), int(h2 * 0.24)
    c0, c1 = int(w2 * 0.17), int(w2 * 0.27)
    alt_recorte, larg_recorte = r1 - r0, c1 - c0
    # Mesma janela em pixels de saída (alt_recorte x larg_recorte) centrada na borda da imagem 8x
    rc_8x = int(((r0 + r1) / 2.0) * 4)
    cc_8x = int(((c0 + c1) / 2.0) * 4)
    r0_8x, r1_8x = rc_8x - alt_recorte // 2, rc_8x + alt_recorte // 2
    c0_8x, c1_8x = cc_8x - larg_recorte // 2, cc_8x + larg_recorte // 2

    fig_zoom, eixos_z = plt.subplots(1, 4, figsize=(16, 4.5))
    fig_zoom.suptitle(
        "Questão 1: Detalhe na Mesma Escala de Pixels — Comparação de Bordas e Interpolação",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )

    eixos_z[0].imshow(fruit1_tam2_2x[r0:r1, c0:c1], interpolation="nearest")
    eixos_z[0].set_title("1.1 tam2 (2x) — Blocos 2x2")
    eixos_z[0].axis("off")

    eixos_z[1].imshow(fruit1_tam2_8x[r0_8x:r1_8x, c0_8x:c1_8x], interpolation="nearest")
    eixos_z[1].set_title("1.1 tam2 (8x) — Blocos 8x8")
    eixos_z[1].axis("off")

    eixos_z[2].imshow(fruit1_tamm_2x[r0:r1, c0:c1], interpolation="nearest")
    eixos_z[2].set_title("1.2 tamm (2x) — Média Bilinear")
    eixos_z[2].axis("off")

    eixos_z[3].imshow(fruit_superres_2x[r0:r1, c0:c1], interpolation="nearest")
    eixos_z[3].set_title("1.3 superres (2x) — Fusão")
    eixos_z[3].axis("off")

    plt.tight_layout()
    caminho_fig2 = pasta_saida / "questao1_zoom_comparacao.png"
    fig_zoom.savefig(caminho_fig2, dpi=150, bbox_inches="tight")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_zoom)

    return {
        "fruit1": fruit1,
        "fruit2": fruit2,
        "tam2_2x": fruit1_tam2_2x,
        "tam2_8x": fruit1_tam2_8x,
        "tamm_2x": fruit1_tamm_2x,
        "superres_2x": fruit_superres_2x,
    }


if __name__ == "__main__":
    executar_questao1(mostrar=True)
