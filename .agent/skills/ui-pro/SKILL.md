---
name: ui-pro
description: Professional UI/UX design and styling skill for Streamlit, web dashboards, and medical AI interfaces. Enforces modern clean aesthetics, glassmorphism, responsive KPI cards, polished charts, and accessible typography.
---

# UI-Pro Skill for Antigravity

This skill defines the professional design system tokens, typography rules, layout hierarchies, and CSS injection patterns for building world-class Streamlit and web user interfaces.

## 1. Design Principles
- **Clarity over Clutter**: Maximize data-ink ratio; avoid unnecessary text and ASCII block diagrams.
- **Visual Hierarchy**: Clear distinction between primary diagnostic indicators, secondary confidence distributions, and background metadata.
- **Clinical Safety Theming**: High-contrast, accessible alert palettes (Emerald for benign/normal, Crimson/Rose for potentially malignant, Amber for warnings).
- **Modern Elevation**: Subtle glassmorphism, soft multi-layered shadows (`box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04)`), and rounded corners (12px to 20px).

## 2. Color System
| Semantic Token | Hex Code | Purpose |
| :--- | :--- | :--- |
| **Primary Navy** | `#091e3a` → `#102a4e` | Hero backgrounds, primary branding |
| **Accent Teal** | `#0d5c63` / `#14b8a6` | Interactive indicators, active state highlights |
| **Card Surface** | `#ffffff` | Elevated data cards, charts |
| **Card Border** | `#e2e8f0` | Subtle structural division |
| **Text Main** | `#0f172a` | Headers, primary data labels |
| **Text Muted** | `#64748b` | Sub-labels, captions, units |
| **Malignant/Risk** | `#ffe4e6` (bg) / `#be123c` (fg) | Potentially malignant status pills |
| **Benign/Normal** | `#d1fae5` (bg) / `#047857` (fg) | Benign lesion status pills |
| **Domain Badge** | `#eff6ff` (bg) / `#1d4ed8` (fg) | Source / Target attribution |

## 3. Typography
- Font Family: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- Title: `2.2rem - 2.4rem`, font-weight `800`, letter-spacing `-0.025em`
- Card Header: `1.15rem - 1.25rem`, font-weight `700`
- KPI Value: `2.0rem - 2.2rem`, font-weight `800`
- Micro Badge: `0.78rem - 0.82rem`, font-weight `700`, uppercase letter-spacing `0.05em`

## 4. Layout Architecture
- **Hero Banner**: Curved gradient header with radial glowing accents and micro-tags.
- **KPI Metrics Row**: 4-column responsive grid with top indicator borders.
- **Side-by-Side Diagnostic Dual Panels**: Contrast comparison cards (Baseline vs Adaptive model).
- **Navigation Sidebar**: Dark contrast theme (`#0f172a`) with live status indicator (`● ENGINE READY`).
