"""
Trabalho 1 - Introdução ao Processamento de Imagens (IPI - UnB)
Script principal de execução integrada das Questões 1 e 2.

Uso:
    python3 main.py            # Executa Questões 1 e 2, salva em resultados/ e exibe janelas
    python3 main.py --no-show  # Executa e salva todas as figuras sem bloquear com plt.show()
"""

from __future__ import annotations

import argparse

from questao1 import executar_questao1
from questao2 import executar_questao2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trabalho 1 - Introdução ao Processamento de Imagens (UnB)"
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="img",
        help="Diretório contendo as imagens de entrada (padrão: img)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resultados",
        help="Diretório onde os gráficos gerados serão salvos (padrão: resultados)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não abrir janelas interativas do Matplotlib (apenas salvar arquivos)",
    )
    args = parser.parse_args()

    mostrar = not args.no_show

    print("=" * 72)
    print("Executando Questão 1: Escalonamento (tam2, tamm) e Super-Resolução")
    print("=" * 72)
    executar_questao1(img_dir=args.img_dir, output_dir=args.output_dir, mostrar=mostrar)

    print("=" * 72)
    print("Executando Questão 2: Realce Power-Law (Gamma) e Equalização de Histograma")
    print("=" * 72)
    executar_questao2(img_dir=args.img_dir, output_dir=args.output_dir, mostrar=mostrar)


if __name__ == "__main__":
    main()
