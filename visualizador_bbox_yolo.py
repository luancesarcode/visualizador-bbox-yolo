#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import deque

import cv2
import numpy as np


PALETA = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def _cor_da_classe(id_classe: str) -> tuple:
    try:
        idx = int(id_classe)
    except ValueError:
        idx = abs(hash(id_classe)) % len(PALETA)
    return PALETA[idx % len(PALETA)]


def _analisar_rotulos(caminho_rotulo: str):
    rotulos = []
    with open(caminho_rotulo, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                print(
                    f"[warn] Linha {line_num} ignorada (esperado >= 5 valores): {line}",
                    file=sys.stderr,
                )
                continue
            id_classe = parts[0]
            try:
                cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                print(
                    f"[warn] Linha {line_num} com valores invalidos: {line}",
                    file=sys.stderr,
                )
                continue
            rotulos.append((id_classe, cx, cy, w, h))
    return rotulos


def _para_xyxy(cx, cy, w, h, img_w, img_h):
    # Heuristica: se todos os valores <= 1, assume YOLO normalizado.
    normalized = max(cx, cy, w, h) <= 1.0
    if normalized:
        cx *= img_w
        cy *= img_h
        w *= img_w
        h *= img_h
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2))
    y2 = int(round(cy + h / 2))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2, normalized


def _desenhar_caixas(imagem, rotulos):
    img_h, img_w = imagem.shape[:2]
    for id_classe, cx, cy, w, h in rotulos:
        x1, y1, x2, y2, normalized = _para_xyxy(cx, cy, w, h, img_w, img_h)
        color = _cor_da_classe(id_classe)
        cv2.rectangle(imagem, (x1, y1), (x2, y2), color, 2)

        texto_rotulo = f"{id_classe}" + (" (norm)" if normalized else " (px)")
        (tw, th), baseline = cv2.getTextSize(
            texto_rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            imagem,
            (x1, max(0, y1 - th - baseline - 4)),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            imagem,
            texto_rotulo,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return imagem


def _highgui_disponivel() -> bool:
    test_window = "__cv2_highgui_test__"
    try:
        cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_window)
        return True
    except cv2.error:
        return False


def _mostrar_com_tkinter(titulo_janela: str, imagem) -> tuple[bool, str | None]:
    try:
        import tkinter as tk

        from PIL import Image, ImageTk
    except Exception as exc:
        return False, f"Falha ao carregar Tkinter/Pillow: {exc}"

    try:
        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((1400, 900))

        root = tk.Tk()
        root.title(f"{titulo_janela} (fallback)")

        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        photo = ImageTk.PhotoImage(pil_image)
        label = tk.Label(frame, image=photo)
        label.image = photo
        label.pack(fill="both", expand=True)

        hint = tk.Label(frame, text="Esc/Q para fechar")
        hint.pack(pady=(8, 0))

        root.bind("<Escape>", lambda _e: root.destroy())
        root.bind("<q>", lambda _e: root.destroy())
        root.bind("<Q>", lambda _e: root.destroy())
        root.mainloop()
        return True, None
    except Exception as exc:
        return False, f"Falha ao exibir imagem com Tkinter: {exc}"


def _anotar_imagem(caminho_imagem, caminho_rotulo, texto_sobreposto=None, permitir_rotulo_ausente=False):
    if not os.path.isfile(caminho_imagem):
        return None, f"Imagem nao encontrada: {caminho_imagem}"

    image = cv2.imread(caminho_imagem)
    if image is None:
        return None, f"Falha ao ler imagem: {caminho_imagem}"

    rotulos = []
    if caminho_rotulo and os.path.isfile(caminho_rotulo):
        rotulos = _analisar_rotulos(caminho_rotulo)
        if not rotulos:
            print("[warn] Nenhuma bbox encontrada no arquivo de rotulos.", file=sys.stderr)
    elif caminho_rotulo and not permitir_rotulo_ausente:
        return None, f"Labels nao encontradas: {caminho_rotulo}"

    annotated = _desenhar_caixas(image, rotulos)
    if texto_sobreposto:
        y = 22
        for line in texto_sobreposto:
            cv2.putText(
                annotated,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 22
    return annotated, None


def _executar_visualizacao(caminho_imagem, caminho_rotulo, caminho_saida=None):
    annotated, err = _anotar_imagem(caminho_imagem, caminho_rotulo)
    if annotated is None:
        return 1, err

    if caminho_saida:
        os.makedirs(os.path.dirname(caminho_saida) or ".", exist_ok=True)
        ok = cv2.imwrite(caminho_saida, annotated)
        if not ok:
            return 1, f"Falha ao salvar imagem: {caminho_saida}"
        print(f"[ok] Imagem salva em: {caminho_saida}")
        return 0, None

    titulo_janela = "bboxes"
    if _highgui_disponivel():
        cv2.imshow(titulo_janela, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0, None

    ok, tk_err = _mostrar_com_tkinter(titulo_janela, annotated)
    if ok:
        print("[warn] OpenCV sem suporte a janela; usando visualizador Tkinter.")
        return 0, None

    return (
        1,
        "OpenCV sem suporte a janela (cv2.imshow). "
        "Use --out para salvar a imagem anotada "
        "ou remova opencv-python-headless da venv. "
        f"Detalhe: {tk_err}",
    )


def _executar_interface():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Visualizar BBox")
    root.geometry("740x320")
    root.minsize(620, 280)

    var_dir_imagens = tk.StringVar()
    var_dir_rotulos = tk.StringVar()
    var_status = tk.StringVar(value="0/0")
    var_arquivo_atual = tk.StringVar(value="-")
    var_status_rotulo = tk.StringVar(value="-")

    lista_imagens = []
    indice_atual = -1
    caminho_estado = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".visualizarbbox_state.json"
    )

    def _carregar_estado_interface():
        if not os.path.isfile(caminho_estado):
            return {}
        try:
            with open(caminho_estado, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            print(f"[warn] Falha ao carregar estado da GUI: {exc}", file=sys.stderr)
        return {}

    def _salvar_estado_interface():
        data = {
            "image_dir": var_dir_imagens.get().strip(),
            "label_dir": var_dir_rotulos.get().strip(),
        }
        try:
            with open(caminho_estado, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[warn] Falha ao salvar estado da GUI: {exc}", file=sys.stderr)

    def _listar_imagens(caminho_diretorio):
        if not caminho_diretorio or not os.path.isdir(caminho_diretorio):
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        arquivos = []
        with os.scandir(caminho_diretorio) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in exts:
                    arquivos.append(entry.path)
        return sorted(arquivos, key=lambda p: os.path.basename(p).lower())

    def _caminho_imagem_atual():
        if lista_imagens and 0 <= indice_atual < len(lista_imagens):
            return lista_imagens[indice_atual]
        return ""

    def _caminho_rotulo_atual():
        caminho_img = _caminho_imagem_atual()
        diretorio_rotulo = var_dir_rotulos.get().strip()
        if not caminho_img or not diretorio_rotulo or not os.path.isdir(diretorio_rotulo):
            return ""
        nome_base = os.path.splitext(os.path.basename(caminho_img))[0]
        return os.path.join(diretorio_rotulo, f"{nome_base}.txt")

    def _nome_rotulo_atual():
        caminho_img = _caminho_imagem_atual()
        diretorio_rotulo = var_dir_rotulos.get().strip()
        if not caminho_img or not diretorio_rotulo or not os.path.isdir(diretorio_rotulo):
            return ""
        nome_base = os.path.splitext(os.path.basename(caminho_img))[0]
        return f"{nome_base}.txt"

    def _atualizar_status():
        if lista_imagens:
            var_status.set(f"{indice_atual + 1}/{len(lista_imagens)}")
        else:
            var_status.set("0/0")

    def _atualizar_rotulos_atuais():
        caminho_img = _caminho_imagem_atual()
        if not caminho_img:
            var_arquivo_atual.set("-")
            var_status_rotulo.set("-")
            return
        var_arquivo_atual.set(os.path.basename(caminho_img))
        caminho_rotulo = _caminho_rotulo_atual()
        nome_rotulo = _nome_rotulo_atual()
        if caminho_rotulo and os.path.isfile(caminho_rotulo):
            var_status_rotulo.set(nome_rotulo or "OK")
        elif caminho_rotulo:
            var_status_rotulo.set(f"{nome_rotulo} (nao encontrado)")
        else:
            var_status_rotulo.set("Sem pasta rotulos")

    def _definir_imagem_atual():
        nonlocal indice_atual
        if not lista_imagens or indice_atual < 0 or indice_atual >= len(lista_imagens):
            _atualizar_status()
            _atualizar_rotulos_atuais()
            return
        _atualizar_status()
        _atualizar_rotulos_atuais()

    def _atualizar_lista_imagens():
        nonlocal lista_imagens, indice_atual
        lista_imagens = _listar_imagens(var_dir_imagens.get().strip())
        if lista_imagens:
            if indice_atual < 0:
                indice_atual = 0
            else:
                indice_atual = min(indice_atual, len(lista_imagens) - 1)
        else:
            indice_atual = -1
        _definir_imagem_atual()

    def _selecionar_imagem():
        pass

    def _selecionar_rotulo():
        pass

    def _aplicar_dir_imagem_do_campo(_event=None):
        _atualizar_lista_imagens()
        _salvar_estado_interface()

    def _aplicar_dir_rotulo_do_campo(_event=None):
        _atualizar_rotulos_atuais()
        _salvar_estado_interface()

    def _selecionar_dir_imagens():
        caminho_pasta = filedialog.askdirectory(title="Selecione a pasta de imagens")
        if caminho_pasta:
            var_dir_imagens.set(caminho_pasta)
            _atualizar_lista_imagens()
            _salvar_estado_interface()

    def _selecionar_dir_rotulos():
        caminho_pasta = filedialog.askdirectory(title="Selecione a pasta de rotulos")
        if caminho_pasta:
            var_dir_rotulos.set(caminho_pasta)
            _atualizar_rotulos_atuais()
            _salvar_estado_interface()

    def _imagem_anterior():
        nonlocal indice_atual
        if not lista_imagens:
            return
        if indice_atual > 0:
            indice_atual -= 1
            _definir_imagem_atual()

    def _imagem_proxima():
        nonlocal indice_atual
        if not lista_imagens:
            return
        if indice_atual < len(lista_imagens) - 1:
            indice_atual += 1
            _definir_imagem_atual()

    def _ao_fechar():
        _salvar_estado_interface()
        root.destroy()

    def _visualizar():
        if lista_imagens and indice_atual >= 0:
            _visualizar_interativo()
            return
        messagebox.showwarning(
            "Campos obrigatorios",
            "Selecione as pastas de imagens e rotulos.",
        )

    def _salvar():
        caminho_imagem = _caminho_imagem_atual()
        caminho_rotulo = _caminho_rotulo_atual()
        if not caminho_imagem:
            messagebox.showwarning(
                "Campos obrigatorios",
                "Selecione a pasta de imagens.",
            )
            return
        out_path = filedialog.asksaveasfilename(
            title="Salvar imagem anotada",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not out_path:
            return
        code, err = _executar_visualizacao(caminho_imagem, caminho_rotulo, out_path)
        if code != 0:
            messagebox.showerror("Erro", err or "Erro desconhecido.")
        else:
            messagebox.showinfo("OK", f"Imagem salva em:\n{out_path}")

    def _abrir_galeria():
        nonlocal indice_atual
        try:
            from PIL import Image, ImageTk
        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Galeria indisponivel (Pillow): {exc}",
            )
            return

        if not lista_imagens:
            messagebox.showwarning(
                "Campos obrigatorios",
                "Selecione a pasta de imagens para abrir a galeria.",
            )
            return

        if hasattr(Image, "Resampling"):
            filtro_reamostragem = Image.Resampling.LANCZOS
        else:
            filtro_reamostragem = Image.LANCZOS

        galeria = tk.Toplevel(root)
        galeria.title("Galeria de imagens")
        galeria.geometry("1100x760")
        galeria.minsize(760, 460)
        galeria.grid_rowconfigure(1, weight=1)
        galeria.grid_columnconfigure(0, weight=1)

        var_status_galeria = tk.StringVar(value="")
        var_selecao = tk.StringVar(value="Selecionada: -")

        barra_superior = ttk.Frame(galeria)
        barra_superior.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=6)
        ttk.Label(barra_superior, textvariable=var_status_galeria).pack(side="left", padx=4)
        ttk.Label(barra_superior, textvariable=var_selecao).pack(side="left", padx=12)
        ttk.Button(barra_superior, text="Abrir selecionada", command=lambda: _abrir_selecionada()).pack(
            side="right",
            padx=4,
        )
        ttk.Button(barra_superior, text="Fechar", command=lambda: _fechar_galeria()).pack(
            side="right", padx=4
        )

        canvas = tk.Canvas(galeria, bg="#1a1a1a", highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        rolagem_y = ttk.Scrollbar(galeria, orient="vertical")
        rolagem_y.grid(row=1, column=1, sticky="ns")
        canvas.configure(yscrollcommand=rolagem_y.set)

        conteiner = ttk.Frame(canvas)
        janela_canvas = canvas.create_window((0, 0), window=conteiner, anchor="nw")

        tamanho_miniatura = (220, 124)
        espacamento_bloco = 6
        colunas = 1
        quantidade_carregada = 0
        job_miniaturas = None
        job_carregar_mais = None
        indice_selecionado = indice_atual if 0 <= indice_atual < len(lista_imagens) else 0
        quadro_selecionado = None
        quadros_blocos = []
        widgets_imagem = []
        fila_miniaturas = deque()
        refs_miniaturas = {}

        miniatura_padrao = ImageTk.PhotoImage(Image.new("RGB", tamanho_miniatura, (42, 42, 42)))
        refs_miniaturas["miniatura_padrao"] = miniatura_padrao

        def _atualizar_status(extra: str | None = None):
            base = f"{len(lista_imagens)} imagens | carregadas {quantidade_carregada}/{len(lista_imagens)}"
            if extra:
                base = f"{base} | {extra}"
            var_status_galeria.set(base)

        def _atualizar_texto_selecao():
            if 0 <= indice_selecionado < len(lista_imagens):
                nome_arquivo = os.path.basename(lista_imagens[indice_selecionado])
                var_selecao.set(
                    f"Selecionada: {indice_selecionado + 1}/{len(lista_imagens)}  {nome_arquivo}"
                )
            else:
                var_selecao.set("Selecionada: -")

        def _destacar_selecionada():
            nonlocal quadro_selecionado
            if quadro_selecionado is not None and quadro_selecionado.winfo_exists():
                quadro_selecionado.configure(highlightbackground="#5a5a5a")
            if 0 <= indice_selecionado < quantidade_carregada:
                quadro_selecionado = quadros_blocos[indice_selecionado]
                if quadro_selecionado.winfo_exists():
                    quadro_selecionado.configure(highlightbackground="#2a7fff")
            else:
                quadro_selecionado = None

        def _selecionar_indice(indice: int):
            nonlocal indice_atual, indice_selecionado
            if not (0 <= indice < len(lista_imagens)):
                return
            indice_selecionado = indice
            indice_atual = indice
            _definir_imagem_atual()
            _atualizar_texto_selecao()
            _destacar_selecionada()

        def _mostrar_galeria():
            if not galeria.winfo_exists():
                return
            galeria.deiconify()
            galeria.lift()
            galeria.focus_set()
            _destacar_selecionada()
            _agendar_talvez_carregar_mais()

        def _abrir_selecionada(_event=None):
            if not (0 <= indice_selecionado < len(lista_imagens)):
                return
            _selecionar_indice(indice_selecionado)
            if not galeria.winfo_exists():
                return

            retornou_do_escape = False

            def _retornar_para_galeria():
                nonlocal retornou_do_escape
                retornou_do_escape = True
                _mostrar_galeria()

            galeria.withdraw()
            _visualizar_interativo(ao_esc=_retornar_para_galeria)
            if galeria.winfo_exists() and not retornou_do_escape:
                _fechar_galeria()

        def _abrir_por_indice(indice: int):
            _selecionar_indice(indice)
            _abrir_selecionada()

        def _criar_miniatura(caminho: str):
            try:
                with Image.open(caminho) as img:
                    rgb = img.convert("RGB")
                    rgb.thumbnail(tamanho_miniatura, filtro_reamostragem)
                    tile_img = Image.new("RGB", tamanho_miniatura, (20, 20, 20))
                    ox = (tamanho_miniatura[0] - rgb.width) // 2
                    oy = (tamanho_miniatura[1] - rgb.height) // 2
                    tile_img.paste(rgb, (ox, oy))
            except Exception:
                tile_img = Image.new("RGB", tamanho_miniatura, (80, 20, 20))
            return ImageTk.PhotoImage(tile_img)

        def _adicionar_bloco(indice: int):
            nome_arquivo = os.path.basename(lista_imagens[indice])
            if len(nome_arquivo) > 44:
                nome_arquivo = f"{nome_arquivo[:41]}..."

            tile = tk.Frame(
                conteiner,
                bg="#1f1f1f",
                highlightthickness=2,
                highlightbackground="#5a5a5a",
            )
            preview = tk.Label(tile, image=miniatura_padrao, bg="#0f0f0f", cursor="hand2")
            preview.image = miniatura_padrao
            preview.pack(fill="both", expand=True)

            caption = tk.Label(
                tile,
                text=nome_arquivo,
                bg="#1f1f1f",
                fg="#f0f0f0",
                anchor="w",
            )
            caption.pack(fill="x", padx=4, pady=(2, 4))

            for widget in (tile, preview, caption):
                widget.bind("<Button-1>", lambda _e, i=indice: _selecionar_indice(i))
                widget.bind("<Double-Button-1>", lambda _e, i=indice: _abrir_por_indice(i))

            quadros_blocos.append(tile)
            widgets_imagem.append(preview)

        def _organizar_blocos():
            nonlocal colunas
            if not galeria.winfo_exists() or quantidade_carregada == 0:
                return
            largura_canvas = max(1, canvas.winfo_width())
            largura_total_bloco = tamanho_miniatura[0] + (espacamento_bloco * 2)
            novas_colunas = max(1, largura_canvas // max(1, largura_total_bloco))
            if novas_colunas != colunas:
                colunas = novas_colunas
                for col in range(colunas):
                    conteiner.grid_columnconfigure(col, weight=1)
            for idx in range(quantidade_carregada):
                frame = quadros_blocos[idx]
                frame.grid(
                    row=idx // colunas,
                    column=idx % colunas,
                    padx=espacamento_bloco,
                    pady=espacamento_bloco,
                    sticky="n",
                )
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        def _sincronizar_regiao_rolagem(_event=None):
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)

        def _enfileirar_miniaturas(indice_inicio: int, indice_fim: int):
            for idx in range(indice_inicio, indice_fim):
                fila_miniaturas.append(idx)
            _agendar_fila_miniaturas()

        def _agendar_fila_miniaturas():
            nonlocal job_miniaturas
            if job_miniaturas is None and galeria.winfo_exists():
                job_miniaturas = galeria.after(1, _processar_fila_miniaturas)

        def _processar_fila_miniaturas():
            nonlocal job_miniaturas
            if not galeria.winfo_exists():
                job_miniaturas = None
                return

            processados = 0
            while fila_miniaturas and processados < 6:
                idx = fila_miniaturas.popleft()
                if not (0 <= idx < quantidade_carregada):
                    continue
                thumb = _criar_miniatura(lista_imagens[idx])
                refs_miniaturas[idx] = thumb
                widget = widgets_imagem[idx]
                if widget.winfo_exists():
                    widget.configure(image=thumb)
                    widget.image = thumb
                processados += 1

            if fila_miniaturas:
                _atualizar_status("gerando miniaturas...")
                job_miniaturas = galeria.after(1, _processar_fila_miniaturas)
            else:
                job_miniaturas = None
                _atualizar_status()

        def _carregar_proximo_lote(tamanho_lote: int = 72):
            nonlocal quantidade_carregada
            if quantidade_carregada >= len(lista_imagens):
                return False
            indice_inicio = quantidade_carregada
            indice_fim = min(len(lista_imagens), indice_inicio + tamanho_lote)
            for idx in range(indice_inicio, indice_fim):
                _adicionar_bloco(idx)
            quantidade_carregada = indice_fim
            _organizar_blocos()
            _destacar_selecionada()
            _enfileirar_miniaturas(indice_inicio, indice_fim)
            _atualizar_status("gerando miniaturas...")
            return True

        def _talvez_carregar_mais(forcar=False):
            if quantidade_carregada >= len(lista_imagens):
                return
            if forcar:
                _carregar_proximo_lote()
                return
            _top, bottom = canvas.yview()
            if bottom >= 0.92:
                _carregar_proximo_lote()

        def _agendar_talvez_carregar_mais():
            nonlocal job_carregar_mais
            if job_carregar_mais is None and galeria.winfo_exists():
                job_carregar_mais = galeria.after(80, _executar_talvez_carregar_mais)

        def _executar_talvez_carregar_mais():
            nonlocal job_carregar_mais
            job_carregar_mais = None
            _talvez_carregar_mais()

        def _ao_redimensionar_canvas(evento):
            canvas.itemconfigure(janela_canvas, width=evento.width)
            _organizar_blocos()
            _agendar_talvez_carregar_mais()

        def _ao_rolar(*argumentos):
            canvas.yview(*argumentos)
            _agendar_talvez_carregar_mais()

        def _ao_roda_mouse(evento):
            if evento.delta:
                canvas.yview_scroll(-int(evento.delta / 120), "units")
            elif getattr(evento, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(evento, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            _agendar_talvez_carregar_mais()

        def _fechar_galeria(_event=None):
            nonlocal job_miniaturas, job_carregar_mais
            if job_miniaturas is not None:
                try:
                    galeria.after_cancel(job_miniaturas)
                except Exception:
                    pass
                job_miniaturas = None
            if job_carregar_mais is not None:
                try:
                    galeria.after_cancel(job_carregar_mais)
                except Exception:
                    pass
                job_carregar_mais = None
            if galeria.winfo_exists():
                galeria.destroy()

        rolagem_y.configure(command=_ao_rolar)
        conteiner.bind("<Configure>", _sincronizar_regiao_rolagem)
        canvas.bind("<Configure>", _ao_redimensionar_canvas)
        galeria.bind("<MouseWheel>", _ao_roda_mouse)
        galeria.bind("<Button-4>", _ao_roda_mouse)
        galeria.bind("<Button-5>", _ao_roda_mouse)
        galeria.bind("<Return>", _abrir_selecionada)
        galeria.bind("<Escape>", _fechar_galeria)
        galeria.protocol("WM_DELETE_WINDOW", _fechar_galeria)

        _atualizar_status()
        _carregar_proximo_lote(tamanho_lote=min(72, len(lista_imagens)))
        _selecionar_indice(indice_selecionado)
        _agendar_talvez_carregar_mais()
        galeria.focus_set()

    def _visualizar_interativo_tkinter(ao_esc=None):
        nonlocal indice_atual
        try:
            from PIL import Image, ImageTk
        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Visualizador fallback indisponivel (Pillow): {exc}",
            )
            return

        if not lista_imagens or indice_atual < 0:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada na pasta.")
            return

        viewer = tk.Toplevel(root)
        viewer.title("bboxes (fallback)")
        viewer.geometry("1100x760")
        viewer.minsize(700, 450)
        viewer.grid_rowconfigure(0, weight=1)
        viewer.grid_rowconfigure(1, weight=0)
        viewer.grid_columnconfigure(0, weight=1)

        var_info = tk.StringVar(value="")
        quadro_imagem = ttk.Frame(viewer)
        quadro_imagem.grid(row=0, column=0, sticky="nsew")
        quadro_imagem.grid_rowconfigure(0, weight=1)
        quadro_imagem.grid_columnconfigure(0, weight=1)

        rotulo_imagem = tk.Label(quadro_imagem, bg="black")
        rotulo_imagem.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(viewer)
        controls.grid(row=1, column=0, sticky="ew")
        ttk.Button(controls, text="Anterior (A)", command=lambda: _anterior_tk()).pack(
            side="left", padx=5
        )
        ttk.Button(controls, text="Proxima (D)", command=lambda: _proxima_tk()).pack(
            side="left", padx=5
        )
        texto_fechar = "Fechar (Q) / Voltar (Esc)" if callable(ao_esc) else "Fechar (Esc/Q)"
        ttk.Button(controls, text=texto_fechar, command=lambda: _fechar_visualizador("close")).pack(
            side="left", padx=5
        )
        ttk.Label(controls, textvariable=var_info).pack(side="left", padx=10)

        indice_cache = None
        imagem_pil_cache = None
        chave_ultima_renderizacao = None
        job_redimensionar = None
        modo_saida = "close"

        if hasattr(Image, "Resampling"):
            reamostrar_reducao = Image.Resampling.LANCZOS
            reamostrar_ampliacao = Image.Resampling.BICUBIC
        else:
            reamostrar_reducao = Image.LANCZOS
            reamostrar_ampliacao = Image.BICUBIC

        def _fechar_visualizador(modo="close"):
            nonlocal modo_saida
            modo_saida = modo
            if viewer.winfo_exists():
                viewer.destroy()

        def _obter_imagem_base():
            nonlocal indice_cache, imagem_pil_cache
            if imagem_pil_cache is not None and indice_cache == indice_atual:
                return imagem_pil_cache, None

            if not lista_imagens or indice_atual < 0:
                return None, "Nenhuma imagem para exibir."
            caminho_img = lista_imagens[indice_atual]
            caminho_rotulo = _caminho_rotulo_atual()
            texto_overlay = [
                f"{indice_atual + 1}/{len(lista_imagens)}  {os.path.basename(caminho_img)}",
                "A/D ou setas: navegar | Q/Esc: sair",
            ]
            annotated, err = _anotar_imagem(
                caminho_img,
                caminho_rotulo,
                texto_sobreposto=texto_overlay,
                permitir_rotulo_ausente=True,
            )
            if annotated is None:
                return None, err or "Erro ao abrir imagem."

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            imagem_pil_cache = Image.fromarray(rgb)
            indice_cache = indice_atual
            return imagem_pil_cache, None

        def _renderizar(forcar=False):
            nonlocal chave_ultima_renderizacao
            if not lista_imagens or indice_atual < 0:
                return

            max_w = rotulo_imagem.winfo_width()
            max_h = rotulo_imagem.winfo_height()
            if max_w < 2 or max_h < 2:
                viewer.after(30, lambda: _renderizar(True))
                return

            chave_renderizacao = (indice_atual, max_w, max_h)
            if not forcar and chave_renderizacao == chave_ultima_renderizacao:
                return

            base_image, err = _obter_imagem_base()
            if base_image is None:
                messagebox.showerror("Erro", err or "Erro ao abrir imagem.")
                _fechar_visualizador("close")
                return

            src_w, src_h = base_image.size
            scale = min(max_w / src_w, max_h / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            if (new_w, new_h) == (src_w, src_h):
                display_image = base_image
            else:
                resample = reamostrar_reducao if scale < 1.0 else reamostrar_ampliacao
                display_image = base_image.resize((new_w, new_h), resample)

            photo = ImageTk.PhotoImage(display_image)
            rotulo_imagem.configure(image=photo)
            rotulo_imagem.image = photo
            var_info.set(f"{indice_atual + 1}/{len(lista_imagens)}")
            _definir_imagem_atual()
            chave_ultima_renderizacao = chave_renderizacao

        def _anterior_tk(_event=None):
            nonlocal indice_atual
            if indice_atual > 0:
                indice_atual -= 1
                _renderizar(True)

        def _proxima_tk(_event=None):
            nonlocal indice_atual
            if indice_atual < len(lista_imagens) - 1:
                indice_atual += 1
                _renderizar(True)

        def _ao_redimensionar(_event=None):
            nonlocal job_redimensionar
            if job_redimensionar is not None:
                try:
                    viewer.after_cancel(job_redimensionar)
                except Exception:
                    pass
            job_redimensionar = viewer.after(40, _renderizar_apos_redimensionar)

        def _renderizar_apos_redimensionar():
            nonlocal job_redimensionar
            job_redimensionar = None
            _renderizar(True)

        viewer.bind("<a>", _anterior_tk)
        viewer.bind("<A>", _anterior_tk)
        viewer.bind("<Left>", _anterior_tk)
        viewer.bind("<d>", _proxima_tk)
        viewer.bind("<D>", _proxima_tk)
        viewer.bind("<Right>", _proxima_tk)
        viewer.bind("<Escape>", lambda _e: _fechar_visualizador("escape"))
        viewer.bind("<q>", lambda _e: _fechar_visualizador("close"))
        viewer.bind("<Q>", lambda _e: _fechar_visualizador("close"))
        viewer.bind("<Configure>", _ao_redimensionar)

        viewer.update_idletasks()
        _renderizar(True)
        viewer.focus_set()
        viewer.wait_window()
        if modo_saida == "escape" and callable(ao_esc):
            ao_esc()

    def _visualizar_interativo(ao_esc=None):
        nonlocal indice_atual
        if not lista_imagens or indice_atual < 0:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada na pasta.")
            return
        if not _highgui_disponivel():
            _visualizar_interativo_tkinter(ao_esc=ao_esc)
            return
        titulo_janela = "bboxes"
        esc_pressionado = False
        while True:
            caminho_img = lista_imagens[indice_atual]
            caminho_rotulo = _caminho_rotulo_atual()
            texto_overlay = [
                f"{indice_atual + 1}/{len(lista_imagens)}  {os.path.basename(caminho_img)}",
                "A/D: navegar | Q/Esc: sair",
            ]
            annotated, err = _anotar_imagem(
                caminho_img,
                caminho_rotulo,
                texto_sobreposto=texto_overlay,
                permitir_rotulo_ausente=True,
            )
            if annotated is None:
                messagebox.showerror("Erro", err or "Erro ao abrir imagem.")
                break
            try:
                cv2.imshow(titulo_janela, annotated)
                key = cv2.waitKey(0)
            except cv2.error:
                cv2.destroyAllWindows()
                _visualizar_interativo_tkinter(ao_esc=ao_esc)
                return
            if key == 27:
                esc_pressionado = True
                break
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                if indice_atual > 0:
                    indice_atual -= 1
                    _definir_imagem_atual()
                continue
            if key in (ord("d"), ord("D")):
                if indice_atual < len(lista_imagens) - 1:
                    indice_atual += 1
                    _definir_imagem_atual()
                continue
        cv2.destroyAllWindows()
        if esc_pressionado and callable(ao_esc):
            ao_esc()

    estado_salvo = _carregar_estado_interface()
    var_dir_imagens.set(str(estado_salvo.get("image_dir", "")).strip())
    var_dir_rotulos.set(str(estado_salvo.get("label_dir", "")).strip())
    _atualizar_lista_imagens()
    _atualizar_rotulos_atuais()

    padding = {"padx": 10, "pady": 8}
    root.columnconfigure(1, weight=1)

    ttk.Label(root, text="Pasta imagens:").grid(row=0, column=0, sticky="w", **padding)
    campo_dir_imagens = ttk.Entry(root, textvariable=var_dir_imagens)
    campo_dir_imagens.grid(row=0, column=1, sticky="ew", **padding)
    ttk.Button(root, text="Selecionar...", command=_selecionar_dir_imagens).grid(
        row=0, column=2, sticky="e", **padding
    )

    ttk.Label(root, text="Pasta rotulos:").grid(row=1, column=0, sticky="w", **padding)
    campo_dir_rotulos = ttk.Entry(root, textvariable=var_dir_rotulos)
    campo_dir_rotulos.grid(row=1, column=1, sticky="ew", **padding)
    ttk.Button(root, text="Selecionar...", command=_selecionar_dir_rotulos).grid(
        row=1, column=2, sticky="e", **padding
    )

    ttk.Label(root, text="Arquivo atual:").grid(row=2, column=0, sticky="w", **padding)
    ttk.Label(root, textvariable=var_arquivo_atual).grid(
        row=2, column=1, columnspan=2, sticky="w", **padding
    )

    ttk.Label(root, text="Label atual:").grid(row=3, column=0, sticky="w", **padding)
    ttk.Label(root, textvariable=var_status_rotulo).grid(
        row=3, column=1, columnspan=2, sticky="w", **padding
    )

    nav_frame = ttk.Frame(root)
    nav_frame.grid(row=4, column=0, columnspan=3, pady=4)
    ttk.Label(nav_frame, textvariable=var_status).pack(side="left", padx=6)

    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
    ttk.Button(btn_frame, text="Visualizar", command=_visualizar).pack(
        side="left", padx=6
    )
    ttk.Button(btn_frame, text="Galeria", command=_abrir_galeria).pack(
        side="left", padx=6
    )
    ttk.Button(btn_frame, text="Salvar imagem", command=_salvar).pack(
        side="left", padx=6
    )
    ttk.Button(btn_frame, text="Sair", command=_ao_fechar).pack(
        side="left", padx=6
    )

    campo_dir_imagens.bind("<Return>", _aplicar_dir_imagem_do_campo)
    campo_dir_imagens.bind("<FocusOut>", _aplicar_dir_imagem_do_campo)
    campo_dir_rotulos.bind("<Return>", _aplicar_dir_rotulo_do_campo)
    campo_dir_rotulos.bind("<FocusOut>", _aplicar_dir_rotulo_do_campo)

    root.bind("a", lambda _e: _imagem_anterior())
    root.bind("A", lambda _e: _imagem_anterior())
    root.bind("d", lambda _e: _imagem_proxima())
    root.bind("D", lambda _e: _imagem_proxima())
    root.protocol("WM_DELETE_WINDOW", _ao_fechar)

    root.mainloop()


def principal():
    analisador = argparse.ArgumentParser(
        description="Visualiza bboxes YOLO em uma imagem."
    )
    analisador.add_argument(
        "caminho_imagem",
        nargs="?",
        help="Caminho da imagem (ex: .jpg/.png)",
    )
    analisador.add_argument(
        "caminho_rotulo",
        nargs="?",
        help="Caminho do .txt YOLO",
    )
    analisador.add_argument(
        "--out",
        dest="caminho_saida",
        default=None,
        help="Salva a imagem anotada nesse caminho (opcional)",
    )
    analisador.add_argument(
        "--gui",
        action="store_true",
        help="Abre a interface grafica para selecionar arquivos",
    )
    argumentos = analisador.parse_args()

    if argumentos.gui or (
        argumentos.caminho_imagem is None and argumentos.caminho_rotulo is None
    ):
        _executar_interface()
        return 0

    if not argumentos.caminho_imagem or not argumentos.caminho_rotulo:
        analisador.print_help()
        return 2

    codigo, erro = _executar_visualizacao(
        argumentos.caminho_imagem,
        argumentos.caminho_rotulo,
        argumentos.caminho_saida,
    )
    if codigo != 0:
        print(f"[erro] {erro}", file=sys.stderr)
    return codigo


if __name__ == "__main__":
    raise SystemExit(principal())


