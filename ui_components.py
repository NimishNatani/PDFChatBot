import streamlit as st

def render_circular_progress_bar(current, max_count):
    percentage = (current / max_count) * 100
    st.markdown(f"""
        <div style="text-align:center;">
            <svg viewBox="0 0 36 36" width="80" height="80">
                <path
                    style="fill:none;stroke:#eee;stroke-width:3.8;"
                    d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                />
                <path
                    style="fill:none;stroke:#4caf50;stroke-width:3.8;stroke-linecap:round;
                    stroke-dasharray:{percentage}, 100;"
                    d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                />
                <text x="18" y="20.35" fill="#4caf50" font-size="7" text-anchor="middle">
                    {current}/{max_count}
                </text>
            </svg>
        </div>
    """, unsafe_allow_html=True)
