"""
Webサイトからコンタクトレンズ情報をスクレイピング

公式サイトやコンタクトレンズ情報サイトからテキストを抽出して
RAG用のドキュメントを作成します。
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import re
import os


def clean_text(text):
    """テキストをクリーンアップ"""
    # 複数の空白を1つに
    text = re.sub(r'\s+', ' ', text)
    # 前後の空白を削除
    text = text.strip()
    return text

def extract_text_from_url(url, max_length=10000):
    """URLからテキストを抽出"""
    try:
        print(f"  取得中: {url}")
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
        
        # テキスト抽出
        text = soup.get_text(separator='\n', strip=True)
        text = clean_text(text)
        
        # 長すぎる場合は切り詰め
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    except Exception as e:
        print(f"  ⚠️  エラー: {e}")
        return None

def scrape_alcon():
    """Alconのサイトからデータ取得"""
    print("\n📥 Alcon のデータを取得中...")
    
    urls = [
        "https://alcon-contact.jp/products/",
        "https://alcon-contact.jp/products/dailies-total1/",
        "https://alcon-contact.jp/products/precision1/",
    ]
    
    all_text = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            all_text.append(f"# {url}\n\n{text}")
        time.sleep(1)  # 礼儀正しくアクセス
    
    if all_text:
        output_file = Path("data/contact_lens_docs/alcon_official.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n---\n\n".join(all_text))
        print(f"  ✅ 保存: {output_file}")
        return True
    return False

def scrape_jnj():
    """J&J (ジョンソン・エンド・ジョンソン) のデータ取得"""
    print("\n📥 J&J のデータを取得中...")
    
    urls = [
        "https://www.jnj.co.jp/",
        "https://acuvue.jnj.co.jp/",
    ]
    
    all_text = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            all_text.append(f"# {url}\n\n{text}")
        time.sleep(1)
    
    if all_text:
        output_file = Path("data/contact_lens_docs/jnj_official.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n\n---\n\n".join(all_text))
        print(f"  ✅ 保存: {output_file}")
        return True
    return False

def scrape_fumufumu():
    """ふむふむ君のデータ取得"""
    print("\n📥 ふむふむ君のデータを取得中...")
    
    # 入力を求める前に環境変数での指定を確認
    env_url = os.getenv("FUMUFUMU_URL", "").strip()
    if env_url:
        url = env_url
        print(f"  環境変数からURLを使用します: {url}")
    else:
        # ふむふむ君のURLを入力してください（例: https://fumufumu-contact.com/）
        url = input("ふむふむ君のURLを入力してください（Enterでスキップ）: ").strip()
    
    if not url:
        print("  ⏭️  スキップしました")
        return False
    
    text = extract_text_from_url(url, max_length=15000)
    if text:
        output_file = Path("data/contact_lens_docs/fumufumu.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# ふむふむ君\n\n{text}")
        print(f"  ✅ 保存: {output_file}")
        return True
    return False

def create_manufacturer_info():
    """主要メーカー情報を手動作成"""
    print("\n📝 メーカー情報を作成中...")
    
    content = """# コンタクトレンズ主要メーカー情報

## アルコン（Alcon）

### 会社概要
スイスに本社を置く眼科医療機器メーカー。コンタクトレンズ分野で世界トップクラスのシェア。

### 主力製品
- **デイリーズ トータル ワン**: 1dayタイプの最高峰。水分含有率が高く、装用感が非常に良い
- **プレシジョン ワン**: コストパフォーマンスに優れた1day
- **エアオプティクス**: 2weekタイプの人気商品

### 特徴
- シリコーンハイドロゲル素材の先駆者
- 酸素透過性が非常に高い
- 乾燥しにくい設計


## ジョンソン・エンド・ジョンソン（J&J）

### 会社概要
アメリカの大手ヘルスケアメーカー。アキュビューブランドで有名。

### 主力製品
- **アキュビュー オアシス**: 高い装用感と乾燥対策
- **アキュビュー モイスト**: 定番の1day
- **アキュビュー トゥルーアイ**: UVカット機能付き

### 特徴
- 世界シェアNo.1
- 豊富な製品ラインナップ
- 初心者にも使いやすい


## HOYA（ホヤ）

### 会社概要
日本の光学機器メーカー。メガネレンズでも有名。

### 主力製品
- **シード 2ウィークピュア**: コストパフォーマンス◎
- **シード 1day Pure**: 国産ブランドの信頼性

### 特徴
- 国内メーカーの安心感
- きめ細やかなサポート
- 日本人の目に合わせた設計


## メニコン

### 会社概要
日本最大のコンタクトレンズメーカー。

### 主力製品
- **メニコンワンデー**: 1dayタイプ
- **メニコンZ**: ハードレンズの代表格
- **メルスプラン**: 定額制サービス

### 特徴
- ハードレンズに強い
- 定額制プラン（メルスプラン）
- 全国に多数の直営店


## ボシュロム

### 会社概要
アメリカの老舗メーカー。150年以上の歴史。

### 主力製品
- **メダリスト**: 2weekタイプの定番
- **バイオトゥルー**: 高含水レンズ

### 特徴
- 長い歴史と実績
- 品質の高さ
- 幅広い製品展開
"""
    
    output_file = Path("data/contact_lens_docs/manufacturers.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✅ 保存: {output_file}")
    return True


def scrape_single_source(url: str, label: str, filename: str, max_length: int = 15000) -> bool:
    """指定URLからテキストを取得して保存"""
    print(f"\n📥 {label} のデータを取得中...")
    text = extract_text_from_url(url, max_length=max_length)
    if not text:
        return False

    output_file = Path("data/contact_lens_docs") / filename
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {label}\n# {url}\n\n{text}")
    print(f"  ✅ 保存: {output_file}")
    return True


def main():
    print("=" * 60)
    print("コンタクトレンズ情報収集スクリプト")
    print("=" * 60)
    
    success_count = 0
    fumufumu_done = False
    
    # メーカー情報を作成
    if create_manufacturer_info():
        success_count += 1
    
    # Alcon
    try:
        if scrape_alcon():
            success_count += 1
    except Exception as e:
        print(f"  ❌ Alcon取得失敗: {e}")
    
    # J&J
    try:
        if scrape_jnj():
            success_count += 1
    except Exception as e:
        print(f"  ❌ J&J取得失敗: {e}")
    
    # ふむふむ君
    try:
        if scrape_fumufumu():
            success_count += 1
            fumufumu_done = True
    except Exception as e:
        print(f"  ❌ ふむふむ君取得失敗: {e}")

    # 追加URL (指定された公式情報)
    if not fumufumu_done:
        try:
            if scrape_single_source(
                "https://contact-tanteidan.seesaa.net/",
                "ふむふむ君（コンタクト探偵団）",
                "fumufumu_contact_tanteidan.txt",
            ):
                success_count += 1
        except Exception as e:
            print(f"  ❌ ふむふむ君 (指定URL) 取得失敗: {e}")

    try:
        if scrape_single_source(
            "https://www.hoya.com/",
            "HOYA 公式サイト",
            "hoya_official.txt",
        ):
            success_count += 1
    except Exception as e:
        print(f"  ❌ HOYA取得失敗: {e}")

    try:
        if scrape_single_source(
            "https://www.eyecity.jp/lp/standard2/?from=eyecity&utm_source=google&utm_medium=cpc&utm_term=&utm_campaign=brand&utm_content=brand_sonota_1&nst=0&gclsrc=aw.ds&&NQBrand1&gad_source=1&gad_campaignid=298117951&gbraid=0AAAAADij2tn02JFSNZQSQPukqgKlxwaj0&gclid=Cj0KCQiA_8TJBhDNARIsAPX5qxQ7fGSY4lQpwEGaFxp9kp3iwKYVRlPfkHFwPrGkwN0lZ2-HYFhcDLkaAkvVEALw_wcB",
            "HOYA アイシティ LP",
            "hoya_eyecity_lp.txt",
        ):
            success_count += 1
    except Exception as e:
        print(f"  ❌ アイシティ取得失敗: {e}")
    
    print("\n" + "=" * 60)
    print(f"完了: {success_count} 件のファイルを作成しました")
    print("=" * 60)
    print("\n次のステップ:")
    print("  python -m src.cli ingest data/contact_lens_docs --output contact_lens_index.pkl")

if __name__ == "__main__":
    main()
