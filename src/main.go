package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

type SteamPriceOverview struct {
	Success     bool   `json:"success"`
	LowestPrice string `json:"lowest_price"`
	Volume      string `json:"volume"`
	MedianPrice string `json:"median_price"`
}

func main() {
	items := []string{
		"AK-47 | Redline (Field-Tested)",
		"AWP | Asiimov (Battle-Scarred)",
		"M4A4 | Neo-Noir (Minimal Wear)",
	}

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	fmt.Println("Penarikan data harian Steam Market...")

	for i, itemName := range items {
		encodedName := url.QueryEscape(itemName)
		
		apiURL := fmt.Sprintf("https://steamcommunity.com/market/priceoverview/?appid=730&currency=10&market_hash_name=%s", encodedName)

		req, err := http.NewRequest("GET", apiURL, nil)
		if err != nil {
			fmt.Printf("[%d/%d] Gagal membuat request untuk %s: %v\n", i+1, len(items), itemName, err)
			continue
		}

		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("[%d/%d] Error jaringan saat menarik %s: %v\n", i+1, len(items), itemName, err)
			continue
		}

		if resp.StatusCode == 429 {
			fmt.Println("Terkena rate limit (429)! hentikan sementara.")
			resp.Body.Close()
			break
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			fmt.Printf("[%d/%d] Gagal menarik %s (Status: %d)\n", i+1, len(items), itemName, resp.StatusCode)
			continue
		}

		var data SteamPriceOverview
		if err := json.Unmarshal(body, &data); err != nil {
			fmt.Printf("[%d/%d] Gagal parsing JSON untuk %s\n", i+1, len(items), itemName)
			continue
		}

		if data.Success {
			fmt.Printf("[%d/%d] BERHASIL | %s\n", i+1, len(items), itemName)
			fmt.Printf("    -> Harga rata-rata: %s | Volume: %s\n", data.MedianPrice, data.Volume)
		} else {
			fmt.Printf("[%d/%d] ITEM TIDAK DITEMUKAN | %s\n", i+1, len(items), itemName)
		}

		if i < len(items)-1 {
			time.Sleep(3 * time.Second)
		}
	}
	
	fmt.Println("Penarikan selesai dijalankan.")
}