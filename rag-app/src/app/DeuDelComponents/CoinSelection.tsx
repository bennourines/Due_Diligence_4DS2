// components/CoinSelection.tsx
import React, { useState } from 'react';
import { Search } from 'lucide-react';

interface CoinSelectionProps {
  onSelectCoin: (coin: string) => void;
}

const CoinSelection: React.FC<CoinSelectionProps> = ({ onSelectCoin }) => {
  const [coinName, setCoinName] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const popularCoins = [
    'Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot', 
    'Ripple', 'Avalanche', 'Chainlink', 'Polygon', 'Cosmos'
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!coinName || coinName.trim() === '') {
      setError('Please enter a cryptocurrency name');
      return;
    }
    
    setError(null);
    onSelectCoin(coinName.trim());
  };

  const handleQuickSelect = (coin: string) => {
    setCoinName(coin);
    onSelectCoin(coin);
  };

  return (
    <div className="flex flex-col min-h-screen bg-deep p-6">
      <header className="mb-8">
        <h1 className="text-2xl font-bold">Cryptocurrency Due Diligence Report</h1>
        <h3 className="text-sm text-secondary mt-2">Start by selecting the cryptocurrency you want to analyze</h3>
      </header>
      
      <div className="flex flex-col items-center justify-center flex-grow">
        <div className="w-full max-w-2xl p-6 border-2 border-[rgba(79,70,229,0.3)] rounded-lg bg-bg-surface">
          <h2 className="text-xl font-semibold mb-6">Select Cryptocurrency</h2>
          
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="flex items-center">
              <div className="relative flex-grow">
                <input
                  type="text"
                  value={coinName}
                  onChange={(e) => setCoinName(e.target.value)}
                  placeholder="Enter cryptocurrency name (e.g., Bitcoin)"
                  className="p-4 w-full pr-10 rounded-lg border-2 border-[rgba(79,70,229,0.3)] bg-transparent text-text-primary focus:border-primary-color outline-none"
                />
                <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-text-secondary" size={20} />
              </div>
              <button
                type="submit"
                className="ml-4 p-4 rounded-lg border-2 bg-primary-color text-white hover:bg-primary-dark"
              >
                Start Analysis
              </button>
            </div>
            {error && <p className="text-red-500 mt-2">{error}</p>}
          </form>
          
          <div>
            <h3 className="text-lg font-medium mb-4">Popular Cryptocurrencies</h3>
            <div className="flex flex-wrap gap-3">
              {popularCoins.map((coin) => (
                <button
                  key={coin}
                  onClick={() => handleQuickSelect(coin)}
                  className="px-4 py-2 rounded-full border-2 border-[rgba(79,70,229,0.3)] hover:border-primary-color"
                >
                  {coin}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CoinSelection;