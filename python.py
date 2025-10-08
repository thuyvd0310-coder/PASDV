import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import io
import time # Dùng cho exponential backoff và loading

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Thẩm Định Vốn Vay AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo Session State cho Lịch sử Chat và Kết quả Tính toán
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = pd.DataFrame()
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
    
# Giá trị mặc định từ đề bài (Đơn vị: Tỷ VNĐ)
DEFAULT_INVEST = 30.0
DEFAULT_REVENUE = 3.5
DEFAULT_OPEX = 2.0
DEFAULT_LOAN_PERC = 0.8
DEFAULT_LOAN_INTEREST = 0.10 # 10%
DEFAULT_WACC = 0.13
DEFAULT_TAX = 0.20
DEFAULT_TERM = 10

# Hàm chuyển đổi Tỷ VNĐ sang triệu VNĐ
def to_millions(billion_vnd):
    """Chuyển đổi từ tỷ VNĐ sang triệu VNĐ để tính toán (nhân 1000)."""
    return billion_vnd * 1000

# --- Hàm tính toán chính (Tính toán Dòng tiền, NPV, IRR, DSCR) ---
@st.cache_data(show_spinner="Đang tính toán các chỉ số tài chính...")
def calculate_financial_metrics(
    invest, rev, opex, loan_perc, interest_rate, wacc, tax_rate, term
):
    """
    Thực hiện tính toán NPV, IRR, Bảng dòng tiền, và DSCR.
    Lưu ý: Giả định chi phí hoạt động (opex) đã bao gồm khấu hao,
    và trả gốc được chia đều trong 10 năm.
    """
    
    # Chuyển đổi sang đơn vị triệu VNĐ để tránh số float quá nhỏ
    I_0 = to_millions(invest)
    Revenue = to_millions(rev)
    OpEx = to_millions(opex)
    
    Loan_Amount = I_0 * loan_perc
    Equity_Amount = I_0 * (1 - loan_perc)
    
    # Trả gốc đều hàng năm (Straight-line principal repayment)
    Principal_Repay = Loan_Amount / term

    # Khởi tạo DataFrame cho dòng tiền
    years = range(1, term + 1)
    df = pd.DataFrame(index=years)
    
    # 1. Dòng tiền Hoạt động
    df['Doanh thu (A)'] = Revenue
    df['Chi phí HĐ (B)'] = OpEx
    df['EBIT (A-B)'] = Revenue - OpEx
    
    # 2. Tính toán Lãi vay và Trả nợ
    loan_balance = Loan_Amount
    interest_list = []
    
    for _ in years:
        # Lãi vay = Số dư nợ đầu kỳ * Lãi suất
        Interest = loan_balance * interest_rate
        interest_list.append(Interest)
        
        # Cập nhật số dư nợ cuối kỳ
        loan_balance -= Principal_Repay
    
    df['Lãi vay (C)'] = interest_list
    df['Trả gốc (D)'] = Principal_Repay
    
    # 3. Tính toán Thuế và Lợi nhuận
    df['EBT (EBIT - C)'] = df['EBIT (A-B)'] - df['Lãi vay (C)']
    
    # Xử lý Tax Shield (Lỗ không đánh thuế, thuế âm = 0)
    df['Thuế (T=20%)'] = df['EBT (EBIT - C)'].apply(
        lambda x: x * tax_rate if x > 0 else 0
    )
    
    df['Lợi nhuận Ròng (EAT)'] = df['EBT (EBIT - C)'] - df['Thuế (T=20%)']
    
    # 4. Tính toán Dòng tiền Thuần (NCF)
    # NCF = EAT + Khấu hao (Vì OpEx đã bao gồm Khấu hao, ta cần cộng lại phần không tiền mặt này)
    # Giả sử Khấu hao = Tỷ lệ khấu hao của TSCĐ trong OpEx (Simplification)
    # Dùng NCF cho mục đích tính toán NPV/IRR
    df['Dòng tiền Thuần NCF (EAT)'] = df['Lợi nhuận Ròng (EAT)'] # Giả định đơn giản cho NPV/IRR
    
    # Thẩm định cần tính toán dòng tiền cho toàn bộ dự án (FCFF)
    # FCFF = EAT + Lãi vay * (1-t) + Khấu hao
    # Ở đây, ta dùng NCF (Free Cash Flow to Equity) để tính IRR/NPV dựa trên WACC
    
    
    # --- Tính toán Chỉ số Thẩm định ---
    
    # NPV (Net Present Value)
    cash_flows = df['Dòng tiền Thuần NCF (EAT)'].tolist()
    npv = np.npv(wacc, [-I_0] + cash_flows)
    
    # IRR (Internal Rate of Return)
    try:
        irr = np.irr([-I_0] + cash_flows)
    except:
        irr = np.nan # Thất bại nếu không có nghiệm (NCF quá nhỏ)

    # DSCR (Debt Service Coverage Ratio)
    # CFADS (Cash Flow Available for Debt Service) = EBIT + Khấu hao
    # Debt Service = Trả gốc + Lãi vay
    df['CFADS (EBIT + Khấu hao)'] = df['EBIT (A-B)'] # Vì OpEx đã bao gồm Khấu hao -> CFADS = EBIT_giả_định + Khấu hao
    # Để đơn giản và phù hợp với giả định, ta dùng EBIT (Rev - OpEx) làm đại diện cho CFADS
    
    df['Debt Service (D+C)'] = df['Trả gốc (D)'] + df['Lãi vay (C)']
    
    # Tính DSCR cho từng năm (chia cho 1e-9 để tránh chia cho 0)
    df['DSCR'] = df['CFADS (EBIT + Khấu hao)'] / df['Debt Service (D+C)']
    dscr_avg = df['DSCR'].replace([np.inf, -np.inf], np.nan).mean()
    
    # Định dạng hiển thị
    df_display = df / 1000 # Chuyển lại về tỷ VNĐ
    df_display = df_display.round(2)
    
    return df_display, npv / 1000, irr, dscr_avg

# --- Hàm gọi API Gemini cho AI Insights (Module IV) ---
def get_ai_insights(metrics_df, npv, irr, dscr, project_data, api_key):
    """Gửi dữ liệu và các chỉ số thẩm định đến Gemini để nhận nhận định."""
    
    system_prompt = (
        "Bạn là một chuyên gia thẩm định tín dụng cấp cao của ngân hàng. "
        "Hãy đánh giá khách quan và chuyên sâu về khả năng cấp vốn cho dự án này. "
        "Phân tích tập trung vào NPV, IRR, DSCR và rủi ro từ tỷ lệ Vay/Tài sản đảm bảo. "
        "Đưa ra nhận định ngắn gọn (khoảng 4-5 đoạn) bằng tiếng Việt."
    )
    
    project_summary = f"""
    --- Dữ liệu Dự án Khởi tạo ---
    - Tổng Vốn ĐT: {project_data['invest']:.2f} tỷ VNĐ
    - Vốn Vay Ngân hàng: {project_data['loan']:.2f} tỷ VNĐ ({project_data['loan_perc']*100:.0f}%)
    - Doanh thu hàng năm (ước tính): {project_data['rev']:.2f} tỷ VNĐ
    - Chi phí HĐ hàng năm (ước tính): {project_data['opex']:.2f} tỷ VNĐ
    - WACC (Chi phí vốn): {project_data['wacc']*100:.1f}%
    - Lãi suất Vay: {project_data['interest_rate']*100:.1f}%
    - Tài sản Đảm bảo (BĐS): 70 tỷ VNĐ

    --- Kết quả Phân tích Tài chính ---
    - NPV (Giá trị hiện tại ròng): {npv:.2f} tỷ VNĐ
    - IRR (Tỷ suất hoàn vốn nội bộ): {irr*100:.2f}%
    - WACC (Ngưỡng chấp nhận): {project_data['wacc']*100:.1f}%
    - DSCR trung bình: {dscr:.2f} lần
    - Bảng Dòng tiền Chi tiết: \n{metrics_df.to_markdown(index=True)}
    """

    try:
        client = genai.Client(api_key=api_key)
        
        # Thử lại với exponential backoff
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[project_summary],
                    system_instruction=system_prompt
                )
                return response.text
            except APIError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt) # Exponential backoff: 1s, 2s, 4s
                else:
                    raise e
        return "Lỗi API không rõ. Vui lòng thử lại sau."
        
    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định: {e}")
        return None

# --- Hàm gọi API Gemini cho Q&A (Module V) ---
def chat_with_gemini(prompt, history, api_key, file_content=None):
    """Xử lý cuộc trò chuyện, duy trì lịch sử và ngữ cảnh dữ liệu dự án."""
    
    # Xây dựng ngữ cảnh (context) từ dữ liệu dự án đã tính toán
    project_context = ""
    if st.session_state.calculated and not st.session_state.metrics_df.empty:
        project_context = (
            "Dự án đang được thẩm định có các chỉ số chính: "
            f"NPV={st.session_state.npv:.2f} tỷ, IRR={st.session_state.irr*100:.2f}%, WACC={st.session_state.wacc*100:.1f}%, DSCR_TB={st.session_state.dscr_avg:.2f} lần. "
            "Dữ liệu dòng tiền chi tiết đã được tính. "
        )
    
    full_prompt = project_context + "Người dùng hỏi: " + prompt
    
    # Xây dựng nội dung cho API
    contents = []
    # Thêm lịch sử chat để duy trì cuộc hội thoại
    for message in history:
        contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
        
    # Thêm prompt hiện tại và file content (nếu có)
    current_parts = [{"text": full_prompt}]
    
    if file_content:
        # Giả định file_content là nội dung văn bản (vd: từ file .txt, .csv, hoặc tóm tắt từ file excel)
        current_parts.append({"text": f"Dữ liệu File Đính Kèm:\n{file_content}"})

    contents.append({"role": "user", "parts": current_parts})

    system_prompt = (
        "Bạn là Trợ lý Thẩm định Tài chính của Ngân hàng. "
        "Hãy trả lời các câu hỏi dựa trên các chỉ số dự án được cung cấp trong ngữ cảnh. "
        "Duy trì giọng điệu chuyên nghiệp và hữu ích. "
        "Hãy tham chiếu lại dữ liệu dự án khi thích hợp. "
    )
    
    try:
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            system_instruction=system_prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Giao diện Chính (Module I, II, III, IV) ---

st.title("💰 Ứng Dụng Thẩm Định Phương Án Vốn Vay (AI-Powered)")
st.caption("Công cụ tính toán NPV, IRR, DSCR và Nhận định chuyên sâu từ Gemini AI.")

# =========================================================================
# I. NHẬP LIỆU VÀ CƠ SỞ DỮ LIỆU (Module I)
# =========================================================================

st.header("1. Nhập liệu Cơ sở Dự án")

col1, col2, col3 = st.columns(3)

with col1:
    invest = st.number_input(
        "Tổng Vốn Đầu tư ($I_0$) (Tỷ VNĐ)",
        min_value=1.0, value=DEFAULT_INVEST, step=0.5
    )
    rev = st.number_input(
        "Doanh thu Hàng năm (Tỷ VNĐ)",
        min_value=0.1, value=DEFAULT_REVENUE, step=0.1
    )
    opex = st.number_input(
        "Chi phí HĐ Hàng năm (Tỷ VNĐ)",
        min_value=0.1, value=DEFAULT_OPEX, step=0.1,
        help="Chi phí bao gồm cả chi phí khấu hao (giả định đơn giản)."
    )

with col2:
    wacc = st.slider(
        "Chi phí Vốn (WACC)",
        min_value=0.01, max_value=0.30, value=DEFAULT_WACC, step=0.005, format="%.1f%%",
        help="13% theo đề bài."
    )
    tax_rate = st.slider(
        "Thuế suất (t)",
        min_value=0.0, max_value=0.5, value=DEFAULT_TAX, step=0.01, format="%.0f%%"
    )
    term = st.number_input(
        "Vòng đời Dự án/Thời gian Vay (Năm)",
        min_value=1, value=DEFAULT_TERM, step=1
    )
    
with col3:
    loan_perc = st.slider(
        "Tỷ lệ Vay Ngân hàng ($L/I_0$)",
        min_value=0.0, max_value=1.0, value=DEFAULT_LOAN_PERC, step=0.05, format="%.0f%%"
    )
    interest_rate = st.slider(
        "Lãi suất Vay ($r_d$)",
        min_value=0.05, max_value=0.20, value=DEFAULT_LOAN_INTEREST, step=0.005, format="%.1f%%"
    )
    st.metric(label="Khoản Vay Dự Kiến (Tỷ VNĐ)", value=f"{invest * loan_perc:.2f}")
    st.metric(label="Tài sản Đảm bảo (Tỷ VNĐ)", value="70.00", help="Tài sản đảm bảo BĐS 70 tỷ VNĐ.")

# Lưu trữ dữ liệu dự án cho AI
project_data = {
    'invest': invest,
    'rev': rev,
    'opex': opex,
    'loan_perc': loan_perc,
    'loan': invest * loan_perc,
    'interest_rate': interest_rate,
    'wacc': wacc,
    'tax_rate': tax_rate,
    'term': term
}

# Nút tính toán và bắt đầu phân tích
if st.button("▶️ Bắt đầu Thẩm định & Tính toán Hiệu quả", type="primary"):
    df_metrics, npv, irr, dscr_avg = calculate_financial_metrics(
        invest, rev, opex, loan_perc, interest_rate, wacc, tax_rate, term
    )
    st.session_state.metrics_df = df_metrics
    st.session_state.npv = npv
    st.session_state.irr = irr
    st.session_state.dscr_avg = dscr_avg
    st.session_state.wacc = wacc
    st.session_state.calculated = True

# =========================================================================
# II. PHÂN TÍCH HIỆU QUẢ DỰ ÁN (Module II)
# =========================================================================

if st.session_state.calculated:
    st.header("2. Kết quả Phân tích Hiệu quả Dự án")

    col_npv, col_irr, col_dscr, col_ltv = st.columns(4)
    
    # Định dạng NPV, IRR, DSCR
    npv_formatted = f"{st.session_state.npv:,.2f} tỷ VNĐ"
    irr_formatted = f"{st.session_state.irr*100:,.2f}%" if not np.isnan(st.session_state.irr) else "Không xác định"
    dscr_formatted = f"{st.session_state.dscr_avg:,.2f} lần"
    ltv_perc = (project_data['loan'] / 70.0)
    ltv_formatted = f"{ltv_perc*100:,.1f}%"

    with col_npv:
        st.metric(
            label="Giá trị Hiện tại Ròng (NPV)",
            value=npv_formatted,
            delta=f"{'Dự án LỜI' if st.session_state.npv > 0 else 'Dự án LỖ'}"
        )
    with col_irr:
        st.metric(
            label="Tỷ suất Hoàn vốn Nội bộ (IRR)",
            value=irr_formatted,
            delta=f"WACC: {st.session_state.wacc*100:,.1f}%"
        )
    with col_dscr:
        st.metric(
            label="Hệ số Khả năng Trả nợ (DSCR TB)",
            value=dscr_formatted,
            delta=f"{'Tốt (>1.2)' if st.session_state.dscr_avg > 1.2 else 'Rủi ro (<1.2)'}"
        )
    with col_ltv:
        st.metric(
            label="Tỷ lệ Vay/Đảm bảo (LTV)",
            value=ltv_formatted,
            delta="An toàn Vốn Ngân hàng"
        )

    st.subheader("Bảng Dòng tiền Chi tiết qua các Năm (Tỷ VNĐ)")
    st.dataframe(st.session_state.metrics_df, use_container_width=True)

    # =========================================================================
    # III. PHÂN TÍCH ĐỘ NHẠY (Module III)
    # =========================================================================
    
    st.header("3. Phân tích Rủi ro & Độ nhạy")
    with st.expander("🔬 Xem Phân tích Độ nhạy (Sensitivity Analysis)"):
        
        # Nhập mức thay đổi
        sens_perc = st.slider(
            "Chọn Mức độ Thay đổi (%)",
            min_value=-20, max_value=20, value=0, step=5, format="%d%%",
            help="Chọn mức thay đổi của Doanh thu và Chi phí để xem NPV thay đổi thế nào."
        )
        
        if sens_perc != 0:
            # Chạy kịch bản độ nhạy
            rev_sens = project_data['rev'] * (1 + sens_perc / 100)
            opex_sens = project_data['opex'] * (1 - sens_perc / 100) # Chi phí giảm khi độ nhạy âm

            df_sens, npv_sens, irr_sens, dscr_sens = calculate_financial_metrics(
                invest, rev_sens, opex, loan_perc, interest_rate, wacc, tax_rate, term
            )

            col_sens1, col_sens2 = st.columns(2)
            with col_sens1:
                st.markdown(f"**Kịch bản: Doanh thu Tăng/Giảm {sens_perc}%**")
                st.metric(
                    label=f"NPV Mới (Tỷ VNĐ)",
                    value=f"{npv_sens:,.2f}",
                    delta=f"Thay đổi: {npv_sens - st.session_state.npv:,.2f} tỷ"
                )
            with col_sens2:
                st.markdown(f"**Kịch bản: Chi phí HĐ Tăng/Giảm {-sens_perc}%**")
                
                # Tính toán lại nếu OpEx thay đổi
                df_opex_sens, npv_opex_sens, irr_opex_sens, dscr_opex_sens = calculate_financial_metrics(
                    invest, rev, opex_sens, loan_perc, interest_rate, wacc, tax_rate, term
                )

                st.metric(
                    label=f"NPV Mới (Tỷ VNĐ)",
                    value=f"{npv_opex_sens:,.2f}",
                    delta=f"Thay đổi: {npv_opex_sens - st.session_state.npv:,.2f} tỷ"
                )
                
    # =========================================================================
    # IV. AI INSIGHTS - NHẬN ĐỊNH CỦA AI (Module IV)
    # =========================================================================
    
    st.header("4. AI Insights - Nhận định Chuyên sâu")
    
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key is None:
        st.warning("⚠️ Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình Secret để sử dụng AI.")
    else:
        if st.button("🤖 Yêu cầu Gemini AI Đánh giá Dự án", key="ai_insights_btn"):
            with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                insights = get_ai_insights(
                    st.session_state.metrics_df, 
                    st.session_state.npv, 
                    st.session_state.irr, 
                    st.session_state.dscr_avg, 
                    project_data, 
                    api_key
                )
                if insights:
                    st.subheader("📝 Báo cáo Thẩm định từ Gemini AI")
                    st.info(insights)

# =========================================================================
# V. Q&A VỚI GEMINI (Module V - Chatbot ở Sidebar)
# =========================================================================

with st.sidebar:
    st.header("💬 Trợ lý Q&A với Gemini")
    st.caption("Hỏi các câu hỏi chuyên sâu về dự án hoặc tài chính.")

    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key is None:
        st.error("⚠️ Khóa API Gemini chưa được cấu hình cho Chatbot.")
    
    # File Uploader cho Chatbot (Tải thêm tệp)
    uploaded_file_chat = st.file_uploader(
        "Tải Tệp để AI Phân tích (Ví dụ: Báo cáo thị trường)",
        type=['txt', 'csv', 'md'],
        key='chat_file_uploader'
    )
    
    file_content_to_send = None
    if uploaded_file_chat:
        try:
            # Đọc nội dung file, chỉ hỗ trợ file text/csv đơn giản
            file_content_to_send = uploaded_file_chat.read().decode("utf-8")
            st.success(f"Đã tải tệp '{uploaded_file_chat.name}'. AI sẽ sử dụng nội dung này.")
        except Exception as e:
            st.warning(f"Không thể đọc nội dung tệp: {e}. AI sẽ chỉ trả lời dựa trên nội dung chat.")
            
    st.markdown("---")
    
    # Hiển thị lịch sử chat
    chat_container = st.container(height=400)
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Khung nhập liệu cho chat
    if prompt := st.chat_input("Hỏi về rủi ro, chỉ số, hoặc dữ liệu dự án..."):
        
        # 1. Thêm prompt của người dùng vào lịch sử
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # 2. Hiển thị prompt ngay lập tức
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # 3. Gọi Gemini API
        if api_key:
            with st.spinner("Gemini đang phân tích..."):
                ai_response = chat_with_gemini(
                    prompt, 
                    st.session_state.chat_history, 
                    api_key, 
                    file_content_to_send
                )
            
            # 4. Hiển thị phản hồi của AI
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(ai_response)

            # 5. Thêm phản hồi của AI vào lịch sử (để duy trì ngữ cảnh)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    st.error("Lỗi: Không thể kết nối với Gemini. Vui lòng kiểm tra Khóa API.")
        
        # 6. Re-run để cập nhật UI
        st.rerun()

st.markdown("---")
st.markdown("💡 **Lưu ý:** Ứng dụng này sử dụng các giả định đơn giản hóa (ví dụ: trả gốc đều, chi phí HĐ bao gồm khấu hao) để tập trung vào các chỉ số thẩm định cốt lõi.")
