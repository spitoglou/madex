#!/usr/bin/env python3
"""
Demonstration of MADEX Analysis with LLM-Friendly Narrative Logging

This script shows how the enhanced sensitivity_analysis.py generates comprehensive narrative
output that is optimized for LLM consumption and interpretation.

Key Features:
1. Structured narrative with clear sections and headers  
2. Clinical context and significance explanations
3. Parameter interpretation with clinical meanings
4. Detailed progress tracking and analysis explanations
5. Results interpretation with clinical implications
6. Comparison analysis between metrics
"""

import os
from pathlib import Path

def main():
    print("MADEX Analysis with LLM-Friendly Narrative Logging")
    print("=" * 60)
    
    print("\n1. Enhanced Logging Features:")
    print("   - Comprehensive narrative output in madex_llm_narrative.log")
    print("   - Clinical scenario explanations with risk levels")
    print("   - Parameter significance and clinical interpretations")
    print("   - Real-time progress tracking with context")
    print("   - Detailed comparison analysis")
    
    print("\n2. LLM-Optimized Output Structure:")
    print("   - Clear section headers and dividers")
    print("   - Contextual explanations for each analysis step")
    print("   - Clinical significance annotations")
    print("   - Quantitative results with qualitative interpretations")
    print("   - Structured summaries and recommendations")
    
    print("\n3. Generated Log Files:")
    log_files = [
        "madex_llm_narrative.log",      # Main LLM-friendly narrative
        "madex_analysis_narrative.log"  # Detailed technical log
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"   [OK] {log_file} ({size:,} bytes)")
        else:
            print(f"   [MISSING] {log_file} (not found)")
    
    print("\n4. Key Narrative Content:")
    print("   - Clinical scenario descriptions with risk assessments")
    print("   - Parameter sensitivity analysis with clinical context")
    print("   - Model comparison results with clinical implications")  
    print("   - Robustness assessment with recommendations")
    print("   - Traditional metric comparison analysis")
    
    print("\n5. Usage for LLMs:")
    print("   - The narrative log provides complete context for analysis")
    print("   - Each section includes both quantitative and qualitative insights")
    print("   - Clinical interpretations make results actionable")
    print("   - Structured format enables easy parsing and understanding")
    
    # Show a sample of the narrative content
    if os.path.exists("madex_llm_narrative.log"):
        print("\n6. Sample Narrative Content:")
        print("-" * 40)
        with open("madex_llm_narrative.log", 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Show first 15 lines as sample
            for i, line in enumerate(lines[:15]):
                print(f"   {line.rstrip()}")
            if len(lines) > 15:
                print(f"   ... ({len(lines) - 15} more lines)")
    
    print(f"\n7. Complete Analysis Available:")
    print("   Run 'python sensitivity_analysis.py' to generate full narrative analysis")
    print("   The madex_llm_narrative.log will contain comprehensive details")

if __name__ == "__main__":
    main()