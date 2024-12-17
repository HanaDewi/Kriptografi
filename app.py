import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# === Fungsi Perhitungan S-Box ===

def validate_sbox(sbox44):
    """
    Validasi apakah S-Box memiliki 256 nilai unik dalam rentang [0-255].
    """
    if len(sbox44) != 256:
        return False, "S-Box harus memiliki tepat 256 nilai."
    if len(set(sbox44)) != 256:
        return False, "S-Box tidak valid: Mengandung nilai duplikat."
    if any(x < 0 or x > 255 for x in sbox44):
        return False, "S-Box mengandung nilai di luar rentang 0-255."
    return True, ""

def calculate_nonlinearity(sbox44):
    """
    Menghitung Nonlinearity S-Box dengan Walsh-Hadamard Transform.
    """
    def walsh_hadamard_transform(f):
        n = 256
        wht = np.zeros(n, dtype=int)
        for a in range(n):
            sum_val = 0
            for x in range(n):
                correlation = (-1) ** (bin(a & x).count('1') % 2)
                sum_val += correlation * ((-1) ** (bin(a & f[x]).count('1') % 2))
            wht[a] = sum_val
        return wht

    wht = walsh_hadamard_transform(sbox44)
    nonlinearity = (256 // 2) - max(abs(wht[1:])) // 2
    return nonlinearity

def calculate_sac(sbox44):
    """
    Menghitung Strict Avalanche Criterion (SAC).
    """
    sac_results = []
    for bit_pos in range(8):
        change_count = sum(
            bin(sbox44[x] ^ sbox44[x ^ (1 << bit_pos)]).count('1')
            for x in range(256)
        )
        sac_results.append(change_count / (256 * 8))
    return np.mean(sac_results)

def calculate_lap(sbox44):
    """
    Menghitung Linear Approximation Probability (LAP).
    """
    max_bias = 0
    for a in range(1, 256):
        for b in range(1, 256):
            correlation_count = sum(
                (bin(a & x).count('1') % 2) == (bin(b & sbox44[x]).count('1') % 2)
                for x in range(256)
            )
            bias = abs(correlation_count / 256.0 - 0.5)
            max_bias = max(max_bias, bias)
    return max_bias

def calculate_dap(sbox44):
    """
    Menghitung Differential Approximation Probability (DAP).
    """
    max_prob = 0
    for delta_in in range(1, 256):
        diff_count = {}
        for x in range(256):
            delta_out = sbox44[x ^ delta_in] ^ sbox44[x]
            diff_count[delta_out] = diff_count.get(delta_out, 0) + 1
        current_max_prob = max(count / 256.0 for count in diff_count.values())
        max_prob = max(max_prob, current_max_prob)
    return max_prob

def calculate_sac_matrix(sbox44):
    """
    Menghitung Strict Avalanche Criterion (SAC) Matrix.
    SAC Matrix berisi probabilitas perubahan bit di output ketika bit input berubah.
    """
    sac_matrix = np.zeros((8, 8))  # 8x8 Matrix untuk setiap posisi bit input-output

    for input_bit in range(8):  # Untuk setiap bit input
        for output_bit in range(8):  # Untuk setiap bit output
            change_count = 0

            for x in range(256):  # Semua kemungkinan input 8-bit
                # Hitung perubahan ketika 1 bit input di-flip
                flipped_x = x ^ (1 << input_bit)
                original_output = (sbox44[x] >> output_bit) & 1
                flipped_output = (sbox44[flipped_x] >> output_bit) & 1

                if original_output != flipped_output:
                    change_count += 1

            # Probabilitas perubahan bit (SAC)
            sac_matrix[input_bit][output_bit] = change_count / 256

    return sac_matrix

# === Streamlit App ===

st.title("S-Box Analysis Tool")
st.markdown("Upload file spreadsheet berisi nilai S-Box untuk dianalisis.")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel S-Box", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)
    st.write("### Data S-Box yang Diunggah:")
    st.dataframe(df)

    # Konversi DataFrame ke list
    sbox44 = df.values.flatten().tolist()

    # Validasi S-Box
    is_valid, error_message = validate_sbox(sbox44)
    if not is_valid:
        st.error(error_message)
    else:
        # Pilihan Operasi
        st.write("### Pilih Operasi Analisis:")
        operations = {
            "Nonlinearity (NL)": calculate_nonlinearity,
            "Strict Avalanche Criterion (SAC)": calculate_sac,
            "Linear Approximation Probability (LAP)": calculate_lap,
            "Differential Approximation Probability (DAP)": calculate_dap,
        }

        selected_ops = st.multiselect("Pilih operasi", options=list(operations.keys()))

        # Jalankan Analisis
        if st.button("Jalankan Analisis"):
            results = {}
            for op in selected_ops:
                results[op] = operations[op](sbox44)

            # Tampilkan Hasil Analisis
            st.write("### Hasil Analisis:")
            results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Nilai"])
            st.dataframe(results_df)

            # Tampilkan SAC Matrix jika SAC dipilih
            if "Strict Avalanche Criterion (SAC)" in selected_ops:
                st.write("### SAC Matrix (Strict Avalanche Criterion):")
                sac_matrix = calculate_sac_matrix(sbox44)
                sac_matrix_df = pd.DataFrame(sac_matrix, 
                                             columns=[f"Output Bit {i}" for i in range(8)],
                                             index=[f"Input Bit {i}" for i in range(8)])
                st.dataframe(sac_matrix_df)

            # Unduh Hasil
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                results_df.to_excel(writer, index=True, sheet_name="Results")
                if "Strict Avalanche Criterion (SAC)" in selected_ops:
                    sac_matrix_df.to_excel(writer, index=True, sheet_name="SAC_Matrix")
            output.seek(0)

            st.download_button(
                label="Unduh Hasil dalam Excel",
                data=output,
                file_name="sbox_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
