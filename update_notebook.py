import json
import os

nb_path = r"c:\Users\wisam\Documents\All_tugas\dataMining\gempa_classify\KlasifikasiGempa\earthquake_clustering.ipynb"

def update_notebook():
    if not os.path.exists(nb_path):
        print(f"Error: File not found at {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    updated_code = False
    updated_md = False

    # New Code Content lines (with \n)
    new_code_source = [
        "# --- ANALISIS CENTROID & PELABELAN BARU ---\n",
        "# Hitung rata-rata fitur tiap klaster (Centroid manual dari data asli)\n",
        "cluster_summary = df.groupby('cluster').agg({\n",
        "    'mag': 'mean',\n",
        "    'depth': 'mean',\n",
        "    'latitude': 'mean', \n",
        "    'longitude': 'mean'\n",
        "}).reset_index()\n",
        "\n",
        "# --- HITUNG EPICENTRAL INTENSITY (GUTENBERG-RICHTER 1956) ---\n",
        "# Formula: I_0 = 1.5 * M - 3.5 * log10(h) + 3.0\n",
        "# Ini estimasi intensitas di episentrum (permukaan)\n",
        "# Depth capped at 5km to avoid singularity at h=0\n",
        "\n",
        "import numpy as np\n",
        "\n",
        "def calculate_intensity_gutenberg_richter(row):\n",
        "    mag = row['mag']\n",
        "    # Cap depth at 5km minimum to avoid singularity\n",
        "    depth = max(row['depth'], 5.0)\n",
        "    \n",
        "    # Gutenberg-Richter (1956) Relation for Epicentral Intensity I_0\n",
        "    # I_0 = 1.5M - 3.5log(h) + 3\n",
        "    intensity = (1.5 * mag) - (3.5 * np.log10(depth)) + 3.0\n",
        "    return intensity\n",
        "\n",
        "cluster_summary['severity_score'] = cluster_summary.apply(calculate_intensity_gutenberg_richter, axis=1)\n",
        "\n",
        "# --- RE-LABELING BERDASARKAN INTENSITY SCORE ---\n",
        "# Urutkan dari score terendah (Low Intensity) ke tertinggi (High Intensity)\n",
        "cluster_summary = cluster_summary.sort_values('severity_score', ascending=True).reset_index(drop=True)\n",
        "\n",
        "# Mapping Label Baru\n",
        "level_names = ['Low', 'Moderate', 'High', 'Very High'] # Asumsi k=4\n",
        "if best_k != 4:\n",
        "    # Fallback jika k bukan 4, generate nama dinamis\n",
        "    level_names = [f'Level {i+1}' for i in range(best_k)]\n",
        "\n",
        "cluster_remap = {}\n",
        "print(\"Cluster ID Remapping (Original -> New Risk Label based on Gutenberg-Richter Intensity):\")\n",
        "\n",
        "for new_id, row in cluster_summary.iterrows():\n",
        "    original_id = int(row['cluster'])\n",
        "    label_name = level_names[min(new_id, len(level_names)-1)]\n",
        "    cluster_remap[original_id] = {\n",
        "        'new_id': new_id,\n",
        "        'label': label_name,\n",
        "        'severity': row['severity_score']\n",
        "    }\n",
        "    print(f\"  Orig Cluster {original_id} -> New ID {new_id} ({label_name}) | Mean Intensity (I0): {row['severity_score']:.2f}\")\n",
        "    \n",
        "    # Logic check warnings\n",
        "    if best_k == 4:\n",
        "        if new_id == 0 and label_name != 'Low':\n",
        "            print(f\"⚠️ WARNING: Cluster {original_id} ranked lowest but assigned {label_name}\")\n",
        "        if new_id == 3 and label_name != 'Very High':\n",
        "            print(f\"⚠️ WARNING: Cluster {original_id} ranked highest but assigned {label_name}\")\n",
        "\n",
        "# Terapkan mapping ke DataFrame utama\n",
        "def apply_remap(c):\n",
        "    if c in cluster_remap:\n",
        "        return cluster_remap[c]['label']\n",
        "    return 'Unknown'\n",
        "\n",
        "df['label'] = df['cluster'].apply(apply_remap)\n",
        "print(\"✅ Labels applied to df['label']\")\n",
        "\n",
        "# Check distribution\n",
        "print(df['label'].value_counts())\n"
    ]

    # Find and update Code Cell
    for cell in cells:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "calculate_weighted_severity" in source_str or "Severity Score dengan Pembobotan" in source_str or "ANALISIS CENTROID" in source_str:
                print("Found Target Code Cell. Updating...")
                cell['source'] = new_code_source
                updated_code = True
                # Clear outputs to avoid confusion
                cell['outputs'] = []
                # break # Continue to find markdown

        elif cell['cell_type'] == 'markdown':
            source_str = "".join(cell['source'])
            if "PENJELASAN LOGIKA RISK CLUSTERING" in source_str or "Weighted Severity Score" in source_str or "0.7" in source_str:
                 # Update Markdown
                 print("Found Target Markdown Cell. Updating...")
                 new_md_source = [
                     "---\n",
                     "## 📚 PENJELASAN LOGIKA RISK CLUSTERING (REVISI SCIENTIFIC)\n",
                     "\n",
                     "### 1. Fitur yang Digunakan\n",
                     "Kita menggunakan 2 fitur fisik utama:\n",
                     "- **Magnitude:** Kekuatan gempa.\n",
                     "- **Depth:** Kedalaman pusat gempa.\n",
                     "\n",
                     "### 2. Gutenberg-Richter Epicentral Intensity ($I_0$)\n",
                     "Untuk menentukan label risiko (Low s/d Very High) secara ilmiah dan mencerminkan **kerusakan di permukaan**, kita menggunakan pendekatan **Intensitas Episentrum** berdasarkan hubungan empiris Gutenberg-Richter (1956):\n",
                     "\n",
                     "$$ I_0 = 1.5 M - 3.5 \\log_{10}(h) + 3.0 $$\n",
                     "\n",
                     "- **$M$ (Magnitude):** Berkontribusi linear positif. Semakin besar M, semakin besar intensitas.\n",
                     "- **$h$ (Depth):** Berkontribusi logaritmik negatif. Semakin dalam gempa, intensitas di permukaan turun drastis (atenuasi).\n",
                     "- **Konstanta 3.0:** Faktor kalibrasi empiris.\n",
                     "\n",
                     "**Kenapa ini Akurat?**\n",
                     "Rumus ini mengoreksi bias rumus linear sebelumnya. Gempa dalam (misal 300km) meskipun Magnitudonya besar, akan menghasilkan $I_0$ yang kecil karena faktor $-3.5 \\log(300)$ yang signifikan (-8.6). Sebaliknya, gempa dangkal (10km) hanya dikurangi sedikit (-3.5), sehingga risikonya tetap tinggi.\n",
                     "\n",
                     "---\n"
                 ]
                 cell['source'] = new_md_source
                 updated_md = True

    if updated_code:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {nb_path}")
    else:
        print("Could not find the target code cell to update.")

if __name__ == "__main__":
    update_notebook()
