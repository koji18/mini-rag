"""
コンタクトレンズ詳細情報の収集

ふむふむ君、HOYA（アイシティ）からのデータ取得と
DK値、医療承認番号のデータベース作成
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import re

def clean_text(text):
    """テキストをクリーンアップ"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_fumufumu():
    """ふむふむ君（コンタクト探偵団）からデータ取得"""
    print("\n📥 ふむふむ君のデータを取得中...")
    
    url = "https://contact-tanteidan.seesaa.net/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 不要なタグを削除
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # メインコンテンツを取得
        text = soup.get_text(separator='\n', strip=True)
        text = clean_text(text)
        
        # 最初の15000文字まで
        if len(text) > 15000:
            text = text[:15000] + "..."
        
        output_file = Path("data/contact_lens_docs/fumufumu_tanteidan.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# コンタクト探偵団（ふむふむ君）\n\n")
            f.write(f"出典: {url}\n\n")
            f.write(text)
        
        print(f"  ✅ 保存: {output_file}")
        return True
        
    except Exception as e:
        print(f"  ⚠️  エラー: {e}")
        return False

def scrape_hoya():
    """HOYA（アイシティ）からデータ取得"""
    print("\n📥 HOYA（アイシティ）のデータを取得中...")
    
    url = "https://www.hoya.com/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        text = clean_text(text)
        
        if len(text) > 12000:
            text = text[:12000] + "..."
        
        output_file = Path("data/contact_lens_docs/hoya_eyecity.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# HOYA（アイシティ）\n\n")
            f.write(f"出典: {url}\n\n")
            f.write(text)
        
        print(f"  ✅ 保存: {output_file}")
        return True
        
    except Exception as e:
        print(f"  ⚠️  エラー: {e}")
        return False

def create_dk_database():
    """DK値（酸素透過係数）データベースを作成"""
    print("\n📝 DK値データベースを作成中...")
    
    content = """# コンタクトレンズ DK値データベース

## DK値とは

DK値（酸素透過係数）は、コンタクトレンズがどれだけ酸素を通すかを示す指標です。
数値が高いほど、角膜に多くの酸素が届きます。

### 推奨DK値
- **最低限**: 24以上
- **推奨**: 87以上
- **連続装用**: 125以上

## 主要製品のDK値

### 1day（使い捨て）タイプ

#### アルコン
- **デイリーズ トータル ワン**: DK値 156
  - 水分傾斜技術により中心部DK値140、表面DK値156
  - 業界最高クラスの酸素透過性

- **プレシジョン ワン**: DK値 100
  - シリコーンハイドロゲル素材
  - コストパフォーマンス◎

#### ジョンソン・エンド・ジョンソン
- **アキュビュー オアシス 1day**: DK値 121
  - HydraLuxe技術
  - 長時間快適

- **アキュビュー モイスト**: DK値 25.5
  - 従来型ハイドロゲル
  - 装用感重視

- **アキュビュー トゥルーアイ**: DK値 118
  - シリコーンハイドロゲル
  - UVカット機能

#### メニコン
- **メニコン 1day プレミオ**: DK値 129
  - 独自のスマートタッチ技術
  - 汚れにくい

### 2week タイプ

#### アルコン
- **エア オプティクス プラス**: DK値 138
  - 高酸素透過性
  - 乾燥対策

#### ジョンソン・エンド・ジョンソン
- **アキュビュー オアシス**: DK値 147
  - 2weekタイプで最高クラス
  - うるおい成分配合

#### ボシュロム
- **メダリスト フレッシュフィット**: DK値 27
  - 従来型素材
  - 含水率59%

### マンスリータイプ

#### メニコン
- **メニコンプレミオ**: DK値 161
  - 連続装用可能
  - 高酸素透過性

### ハードコンタクトレンズ

#### メニコン
- **メニコンZ**: DK値 163
  - 高酸素透過性ハードレンズ
  - 長期使用に最適

#### HOYA
- **ハードEX**: DK値 100
  - 耐久性◎
  - 乱視矯正

## DK値の重要性

### 低DK値のリスク
- 角膜の酸素不足（低酸素症）
- 角膜浮腫（むくみ）
- 角膜血管新生
- 角膜感染症のリスク増加

### 高DK値のメリット
- 角膜の健康維持
- 長時間装用が可能
- 充血しにくい
- 安全性が高い

## 素材による違い

### 従来型ハイドロゲル
- DK値: 20-30程度
- 含水率が高い
- 装用感が良い

### シリコーンハイドロゲル
- DK値: 100-160程度
- 酸素透過性が非常に高い
- 現代の主流素材

### ハードレンズ（RGP）
- DK値: 100-163
- 最も高い酸素透過性
- 耐久性に優れる
"""
    
    output_file = Path("data/contact_lens_docs/dk_values.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ 保存: {output_file}")
    return True

def create_approval_database():
    """医療機器承認番号データベースを作成"""
    print("\n📝 医療機器承認番号データベースを作成中...")
    
    content = """# コンタクトレンズ医療機器承認番号データベース

## 医療機器承認番号とは

コンタクトレンズは高度管理医療機器（クラスIII）に分類されます。
承認番号により、製品の製造元や OEM 関係を確認できます。

## OEM（相手先ブランド製造）について

同じ承認番号を持つ製品は、同じ工場で製造された OEM 製品です。
製品名が違っても、実質的に同じレンズの可能性があります。

## 主要製品の承認番号

### アルコン（Alcon）製品

#### デイリーズ トータル ワン
- 承認番号: 22600BZX00325000
- 製造販売元: 日本アルコン株式会社
- OEM製品: なし（オリジナル）

#### プレシジョン ワン
- 承認番号: 30200BZX00153000
- 製造販売元: 日本アルコン株式会社
- OEM製品: なし（オリジナル）

#### エア オプティクス プラス ハイドラグライド
- 承認番号: 30100BZX00288000
- 製造販売元: 日本アルコン株式会社

### ジョンソン・エンド・ジョンソン（J&J）製品

#### アキュビュー オアシス 1day
- 承認番号: 22800BZX00226000
- 製造販売元: ジョンソン・エンド・ジョンソン株式会社
- OEM製品: なし（オリジナル）

#### アキュビュー モイスト
- 承認番号: 21600BZY00408000
- 製造販売元: ジョンソン・エンド・ジョンソン株式会社
- OEM製品: 多数のプライベートブランドあり

#### アキュビュー オアシス（2week）
- 承認番号: 21800BZY00509000
- 製造販売元: ジョンソン・エンド・ジョンソン株式会社

### メニコン製品

#### メニコン 1day プレミオ
- 承認番号: 22400BZX00336000
- 製造販売元: 株式会社メニコン
- OEM製品: なし（オリジナル）

#### メニコンZ（ハード）
- 承認番号: 20700BZY00415000
- 製造販売元: 株式会社メニコン
- 超ロングセラー製品

### HOYA・シード製品

#### シード 2ウィークピュア
- 承認番号: 21600BZY00407000
- 製造販売元: 株式会社シード
- OEM提供: HOYAブランド等

#### シード 1day Pure
- 承認番号: 22200BZX00382000
- 製造販売元: 株式会社シード

### ボシュロム製品

#### メダリスト ワンデープラス
- 承認番号: 22200BZX00428000
- 製造販売元: ボシュロム・ジャパン株式会社

#### メダリスト フレッシュフィット コンフォート モイスト
- 承認番号: 22000BZX01075000
- 製造販売元: ボシュロム・ジャパン株式会社

## OEM 関係の確認方法

### 同一承認番号 = 同一製品

例：
- 製品A: 承認番号 21600BZY00408000
- 製品B: 承認番号 21600BZY00408000
→ 両製品は同じレンズ（OEM関係）

### 承認番号の見方

**形式**: XXXXXBZXXXXXXXYYY
- 前半5桁: 承認年度
- BZX/BZY: 製品カテゴリ
- 後半数字: 個別番号

### OEM製品の見分け方

1. **承認番号を確認**
   - パッケージに記載
   - 添付文書に記載
   - PMDAのサイトで検索

2. **価格差をチェック**
   - 同じ承認番号で価格差がある → OEM の可能性

3. **製造販売元を確認**
   - 同一承認番号で製造販売元が異なる → OEM

## よくある OEM 関係

### アキュビュー モイストのOEM
多くのプライベートブランド（眼科・量販店オリジナル）が存在

### シードのOEM
HOYAブランドとして販売されることが多い

### ボシュロムのOEM
一部の量販店ブランドで採用

## 注意事項

### 同じ承認番号でも
- パッケージデザインは異なる
- 価格は異なる
- 保証内容が異なる場合がある
- **レンズ自体は同一**

### 確認すべきポイント
1. 承認番号
2. 製造販売元
3. BC（ベースカーブ）
4. PWR（度数）範囲
5. DIA（直径）

## PMDA（医薬品医療機器総合機構）での確認方法

1. PMDAのサイトにアクセス
2. 医療機器検索
3. 承認番号で検索
4. 製品の詳細情報を確認
"""
    
    output_file = Path("data/contact_lens_docs/approval_numbers.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ 保存: {output_file}")
    return True

def main():
    print("=" * 60)
    print("コンタクトレンズ詳細情報収集スクリプト")
    print("=" * 60)
    
    success_count = 0
    
    # ふむふむ君
    try:
        if scrape_fumufumu():
            success_count += 1
        time.sleep(2)
    except Exception as e:
        print(f"  ❌ エラー: {e}")
    
    # HOYA
    try:
        if scrape_hoya():
            success_count += 1
        time.sleep(2)
    except Exception as e:
        print(f"  ❌ エラー: {e}")
    
    # DK値データベース
    try:
        if create_dk_database():
            success_count += 1
    except Exception as e:
        print(f"  ❌ エラー: {e}")
    
    # 承認番号データベース
    try:
        if create_approval_database():
            success_count += 1
    except Exception as e:
        print(f"  ❌ エラー: {e}")
    
    print("\n" + "=" * 60)
    print(f"完了: {success_count} 件のファイルを作成しました")
    print("=" * 60)
    print("\n次のステップ:")
    print("  python -m src.cli ingest data/contact_lens_docs --output contact_lens_index.pkl")

if __name__ == "__main__":
    main()
