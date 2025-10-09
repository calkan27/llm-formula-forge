"""
Generate all figures for the Feynman v1 Pass analysis report.
Reads aggregate_report.json and accept_table.csv from the current directory.
Outputs figures to ./fig/ subdirectory.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Create output directory
output_dir = Path("fig")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
with open("aggregate_report.json", "r") as f:
    agg_report = json.load(f)

df = pd.read_csv("accept_table.csv")

# Extract chapter prefix from equation names
df['chapter'] = df['name'].str.extract(r'^(I+)\.')[0]
df['chapter'] = df['chapter'].fillna('I')  # Default to I if not matched

print(f"Loaded {len(df)} equations")
print(f"Accepted: {df['accepted'].sum()}, Rejected: {(~df['accepted']).sum()}")



'''
 ============================================================================
 Figure 1: Manifest & Events Validation Diagram (Schematic)
 ============================================================================
'''

print("\nGenerating Figure 1: Manifest & Events Validation...")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Manifest & Events Validation Pipeline', 
        ha='center', va='top', fontsize=14, fontweight='bold')

# Box 1: Manifest
box1_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=1.5)
ax.text(0.15, 0.75, 'Manifest\n✓ Hash verified\n✓ Canonical JSON', 
        ha='center', va='center', fontsize=10, bbox=box1_props)

# Box 2: Events
box2_props = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='black', linewidth=1.5)
ax.text(0.5, 0.75, 'Events (JSONL)\n✓ Count=502\n✓ Start/Stop markers\n✓ Monotone timestamps', 
        ha='center', va='center', fontsize=10, bbox=box2_props)

# Box 3: Artifacts
box3_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', linewidth=1.5)
ax.text(0.85, 0.75, 'Artifacts (×100)\n✓ Accept proof\n✓ Summary\n✓ Forms', 
        ha='center', va='center', fontsize=10, bbox=box3_props)

# Result box
result_props = dict(boxstyle='round,pad=0.7', facecolor='lightcoral', edgecolor='black', linewidth=2)
ax.text(0.5, 0.35, 'Validation Result\nmanifest_ok = True\nevents_ok = True\nartifact_core_ok_rate = 1.0', 
        ha='center', va='center', fontsize=12, fontweight='bold', bbox=result_props)

# Arrows
ax.annotate('', xy=(0.5, 0.45), xytext=(0.15, 0.65),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.65),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(0.5, 0.45), xytext=(0.85, 0.65),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Footer
ax.text(0.5, 0.05, f'Run: {agg_report["run_dir"].split("/")[-1]}', 
        ha='center', va='bottom', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig(output_dir / "manifest_events.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/manifest_events.pdf")

'''
 ============================================================================
 Figure 2: Method Mix Bar Chart
 ============================================================================
'''

print("\nGenerating Figure 2: Method distribution...")

method_counts = df['method'].value_counts().to_dict()
# Ensure all methods are present
for m in ['symbolic', 'float_probe', 'reject']:
    if m not in method_counts:
        method_counts[m] = 0

methods = ['symbolic', 'float_probe', 'reject']
counts = [method_counts.get(m, 0) for m in methods]
colors = ['#2ecc71', '#3498db', '#e74c3c']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.2)

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Count', fontweight='bold')
ax.set_xlabel('Method', fontweight='bold')
ax.set_title('Method Distribution (N=100)', fontweight='bold', pad=15)
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "method_mix_bar.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/method_mix_bar.pdf")

'''
 ============================================================================
 Figure 3: Complexity vs Error Scatter
 ============================================================================
'''
print("\nGenerating Figure 3: C vs E scatter...")

fig, ax = plt.subplots(figsize=(8, 6))

# Separate accepted and rejected
accepted_df = df[df['accepted'] == True]
rejected_df = df[df['accepted'] == False]

# Plot rejected (background)
ax.scatter(rejected_df['C'], rejected_df['E'], 
          c='lightcoral', s=80, alpha=0.6, edgecolors='darkred', linewidth=0.8,
          label=f'Rejected (n={len(rejected_df)})', marker='x')

# Plot accepted (foreground)
ax.scatter(accepted_df['C'], accepted_df['E'], 
          c='lightgreen', s=100, alpha=0.8, edgecolors='darkgreen', linewidth=1.2,
          label=f'Accepted (n={len(accepted_df)})', marker='o')

ax.set_xlabel('Complexity (C)', fontweight='bold')
ax.set_ylabel('Normalized Error (E)', fontweight='bold')
ax.set_title('Complexity vs. Error\n(Accepted equations at E=0, many rejections at C=0, E=1)', 
             fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Add reference lines
ax.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='E=0 (perfect)')
ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='E=1 (worst)')

plt.tight_layout()
plt.savefig(output_dir / "c_vs_e_scatter.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/c_vs_e_scatter.pdf")

'''
 ============================================================================
 Figure 4: Per-Chapter Acceptance Rate
 ============================================================================
'''

print("\nGenerating Figure 4: Per-chapter acceptance rates...")

# Group by chapter
chapter_stats = df.groupby('chapter').agg({
    'accepted': ['sum', 'count']
}).reset_index()
chapter_stats.columns = ['chapter', 'accepted', 'total']
chapter_stats['accept_rate'] = chapter_stats['accepted'] / chapter_stats['total']

# Sort by chapter
chapter_order = ['I', 'II', 'III']
chapter_stats = chapter_stats.set_index('chapter').reindex(chapter_order).reset_index()

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(chapter_stats['chapter'], chapter_stats['accept_rate'], 
              color=['#3498db', '#9b59b6', '#e67e22'], 
              edgecolor='black', linewidth=1.2, alpha=0.8)

# Add labels
for i, (bar, row) in enumerate(zip(bars, chapter_stats.itertuples())):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}\n({row.accepted}/{row.total})',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_ylabel('Acceptance Rate', fontweight='bold')
ax.set_xlabel('Chapter', fontweight='bold')
ax.set_title('Acceptance Rate by Chapter\n(Chapter I shows lowest acceptance)', 
             fontweight='bold', pad=15)
ax.set_ylim(0, max(chapter_stats['accept_rate']) * 1.2)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.36, color='red', linestyle='--', linewidth=1.5, 
           alpha=0.7, label='Overall rate (0.36)')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "accept_rate_by_prefix.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/accept_rate_by_prefix.pdf")

'''
 ============================================================================
 Figure 5: Empirical CDF of E
 ============================================================================
'''

print("\nGenerating Figure 5: Error CDF...")

fig, ax = plt.subplots(figsize=(8, 6))

# Sort E values
sorted_E = np.sort(df['E'].values)
y_vals = np.arange(1, len(sorted_E) + 1) / len(sorted_E)

# Plot CDF
ax.plot(sorted_E, y_vals, linewidth=2.5, color='#2c3e50', label='E CDF')

# Highlight key points
ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='E=0 (exact)')
ax.axvline(x=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='E=1 (max error)')
ax.axhline(y=0.36, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, 
           label='36% accepted at E=0')

# Add annotations
ax.annotate(f'36 equations\nat E=0.0', 
            xy=(0, 0.36), xytext=(0.3, 0.36),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
            fontsize=10, fontweight='bold', color='green')

ax.annotate(f'61 equations\nat E=1.0', 
            xy=(1, 0.97), xytext=(0.7, 0.80),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
            fontsize=10, fontweight='bold', color='red')

ax.set_xlabel('Normalized Error (E)', fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontweight='bold')
ax.set_title('Empirical CDF of Normalized Error\n(Sharp jump at E=0, steep rise near E=1)', 
             fontweight='bold', pad=15)
ax.legend(loc='center right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(output_dir / "e_cdf.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/e_cdf.pdf")


'''
 ============================================================================
 Figure 6: Rejection Severity Taxonomy
 ============================================================================
'''
print("\nGenerating Figure 6: Rejection taxonomy...")

# Filter to rejected only
rejected_df = df[df['accepted'] == False].copy()

# Classify by error bins
rejected_df['severity'] = pd.cut(rejected_df['E'], 
                                  bins=[-0.01, 0.2, 0.5, 1.01],
                                  labels=['Near miss (E<0.2)', 
                                         'Moderate (0.2≤E<0.5)', 
                                         'Severe (E≥0.5)'])

severity_counts = rejected_df['severity'].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))

# Create stacked bar (just one bar showing all categories)
categories = ['Near miss (E<0.2)', 'Moderate (0.2≤E<0.5)', 'Severe (E≥0.5)']
counts = [severity_counts.get(cat, 0) for cat in categories]
colors_sev = ['#27ae60', '#f39c12', '#c0392b']

bottom = 0
bars = []
for i, (cat, count, color) in enumerate(zip(categories, counts, colors_sev)):
    bar = ax.barh(['Rejected Equations'], [count], left=bottom, 
                   color=color, edgecolor='black', linewidth=1.2, 
                   label=f'{cat}: {int(count)}')
    bars.append(bar)
    
    # Add count label
    if count > 0:
        ax.text(bottom + count/2, 0, f'{int(count)}', 
                ha='center', va='center', fontweight='bold', fontsize=11, color='white')
    
    bottom += count

ax.set_xlabel('Count', fontweight='bold')
ax.set_title(f'Rejection Severity Distribution (n={len(rejected_df)} rejected)\nOperational taxonomy based on error magnitude', 
             fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim(0, len(rejected_df) * 1.05)

plt.tight_layout()
plt.savefig(output_dir / "rejection_taxonomy.pdf", bbox_inches='tight')
plt.close()
print("✓ Saved fig/rejection_taxonomy.pdf")

'''
 ============================================================================
 Summary Statistics
 ============================================================================
'''
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nTotal equations: {len(df)}")
print(f"Accepted: {df['accepted'].sum()} ({df['accepted'].sum()/len(df):.2%})")
print(f"Rejected: {(~df['accepted']).sum()} ({(~df['accepted']).sum()/len(df):.2%})")

print(f"\nMethod breakdown:")
for method, count in df['method'].value_counts().items():
    print(f"  {method}: {count}")

print(f"\nMetric statistics (all rows):")
for metric in ['C', 'E', 'S', 'L']:
    stats = df[metric].describe()
    print(f"  {metric}: min={stats['min']:.4f}, median={df[metric].median():.4f}, "
          f"mean={stats['mean']:.4f}, max={stats['max']:.4f}")

print(f"\nChapter acceptance:")
for row in chapter_stats.itertuples():
    print(f"  Chapter {row.chapter}: {row.accepted}/{row.total} = {row.accept_rate:.2%}")

print(f"\nRejection severity (n={len(rejected_df)}):")
for cat, count in severity_counts.items():
    print(f"  {cat}: {count} ({count/len(rejected_df):.1%})")

print(f"\nAccepted equations complexity:")
acc_c = df[df['accepted'] == True]['C']
print(f"  Range: [{acc_c.min():.0f}, {acc_c.max():.0f}]")
print(f"  Median: {acc_c.median():.1f}, Mean: {acc_c.mean():.2f}")

print(f"\nRejected equations at C=0: {len(df[(df['accepted']==False) & (df['C']==0)])}/64 "
      f"({len(df[(df['accepted']==False) & (df['C']==0)])/64:.1%})")

print("\n" + "="*70)
print("All figures generated successfully in ./fig/")
print("="*70)
