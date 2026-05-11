const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const apiStatusBadge = document.getElementById('apiStatus');
const statusText = document.getElementById('statusText');
const tickerValue = document.getElementById('tickerValue');
const lastClosePriceEl = document.getElementById('lastClosePrice');
const lastCloseDateEl = document.getElementById('lastCloseDate');
const predictedPriceEl = document.getElementById('predictedPrice');
const predictionFooter = document.getElementById('predictionFooter');
const predictBtn = document.getElementById('predictBtn');
const predictDaysInput = document.getElementById('predictDays');
const downloadBtn = document.getElementById('downloadBtn');
const refreshBtn = document.getElementById('refreshBtn');
const chartLoader = document.getElementById('chartLoader');

let stockChart;
let currentHistoryData = [];
let currentForecastData = [];

// Check API connection health
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/`);
        if (res.ok) {
            apiStatusBadge.classList.remove('error');
            apiStatusBadge.classList.add('online');
            statusText.innerText = 'Connected to API';
            return true;
        }
    } catch (e) {
        console.error("Backend inaccessible:", e);
    }
    apiStatusBadge.classList.remove('online');
    apiStatusBadge.classList.add('error');
    statusText.innerText = 'Disconnected';
    return false;
}

// Format money value
function formatCurrency(value) {
    return new Intl.NumberFormat('id-ID', {
        style: 'currency',
        currency: 'IDR',
        minimumFractionDigits: 0
    }).format(value);
}

// Initialize Chart
function initChart(labels, data) {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    let gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 188, 212, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 188, 212, 0.0)');

    if (stockChart) {
        stockChart.destroy();
    }

    Chart.defaults.color = '#8b9bb4';
    Chart.defaults.font.family = "'Outfit', sans-serif";

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Historical Close',
                data: data,
                borderColor: '#00bcd4',
                backgroundColor: gradient,
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#00bcd4',
                pointHoverBorderColor: '#fff',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(9, 11, 16, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: (context) => `Price: ${formatCurrency(context.parsed.y)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: { maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    ticks: {
                        callback: (value) => value.toLocaleString('id-ID')
                    }
                }
            }
        }
    });
}

// Add projections range to the chart
function updateChartWithPredictions(forecastPoints) {
    if (!stockChart || !window.currentLabels || !window.currentPrices) return;

    const histLabels = [...window.currentLabels];
    const histPrices = [...window.currentPrices];

    const futureLabels = forecastPoints.map(p => p.date);
    const futurePrices = forecastPoints.map(p => p.price);

    // Combined all labels for X Axis
    const combinedLabels = [...histLabels, ...futureLabels];

    // The projection dataset starts with nulls equal to length-1 of history, then connects the final point to the start of projections
    const forecastValues = new Array(histPrices.length - 1).fill(null);
    forecastValues.push(histPrices[histPrices.length - 1]); // Last historical point connection
    futurePrices.forEach(val => forecastValues.push(val));

    stockChart.data.labels = combinedLabels;

    if (stockChart.data.datasets.length > 1) {
        stockChart.data.datasets[1].data = forecastValues;
    } else {
        stockChart.data.datasets.push({
            label: 'AI Forecast',
            data: forecastValues,
            borderColor: '#00e676',
            borderDash: [5, 5],
            borderWidth: 3,
            tension: 0.3,
            pointRadius: 4,
            pointBackgroundColor: '#00e676',
            fill: false
        });
    }
    stockChart.update();
}

// Fetch history
async function loadDashboardData() {
    chartLoader.classList.add('active');
    downloadBtn.disabled = true;
    const connected = await checkHealth();
    if (!connected) {
        chartLoader.classList.remove('active');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=60`);
        const result = await response.json();
        
        if (result.history && result.history.length > 0) {
            currentHistoryData = result.history; // Save raw records
            
            const labels = currentHistoryData.map(d => d.date);
            const prices = currentHistoryData.map(d => d.close);
            
            const lastEntry = currentHistoryData[currentHistoryData.length - 1];
            
            tickerValue.innerText = result.ticker || 'BBCA.JK';
            lastClosePriceEl.innerText = formatCurrency(lastEntry.close);
            lastCloseDateEl.innerText = `As of ${lastEntry.date}`;
            
            initChart(labels, prices);
            
            window.currentLabels = labels;
            window.currentPrices = prices;

            currentForecastData = []; // clear forecasts reset
        }
    } catch (e) {
        console.error("Failed to fetch history", e);
    } finally {
        chartLoader.classList.remove('active');
    }
}

// Run predictions
async function runPrediction() {
    predictBtn.disabled = true;
    downloadBtn.disabled = true;
    const originalText = predictBtn.innerHTML;
    predictBtn.innerHTML = '<div class="spinner" style="width:15px;height:15px;border-width:2px;margin-bottom:0"></div><span>Computing...</span>';
    
    // Fetch chosen number of days
    const days = parseInt(predictDaysInput.value) || 1;

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ days: days })
        });
        
        if (!res.ok) {
            const errTxt = await res.text();
            throw new Error(errTxt);
        }
        
        const data = await res.json();
        currentForecastData = data.predictions; 
        
        // Update summary with the target (final day) forecast value
        const finalPoint = currentForecastData[currentForecastData.length - 1];
        predictedPriceEl.innerText = formatCurrency(finalPoint.price);
        
        const daysStr = days === 1 ? 'Tomorrow' : `in ${days} Business Days`;
        predictionFooter.innerHTML = `<span style="color:#00e676"><i class="fa-solid fa-circle-check"></i> Est. ${daysStr}</span>`;
        
        // Visualise on chart
        updateChartWithPredictions(currentForecastData);
        
        // Enable download after successful creation
        downloadBtn.disabled = false;

    } catch (e) {
        console.error(e);
        predictedPriceEl.innerText = "Error";
        predictionFooter.innerText = "Failed forecast.";
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = originalText;
    }
}

// CSV Downloader
function downloadCSV() {
    if (!currentHistoryData.length) return;

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Date,Type,Price\n";

    // Append history
    currentHistoryData.forEach(item => {
        csvContent += `${item.date},Historical,${item.close}\n`;
    });

    // Append forecasts
    currentForecastData.forEach(item => {
        csvContent += `${item.date},AI Forecast,${item.price.toFixed(2)}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    
    const d = new Date().toISOString().split('T')[0];
    link.setAttribute("download", `stock_forecast_export_${d}.csv`);
    
    document.body.appendChild(link); // Needed for Firefox
    link.click();
    document.body.removeChild(link);
}

// Event Listeners
predictBtn.addEventListener('click', runPrediction);
refreshBtn.addEventListener('click', loadDashboardData);
downloadBtn.addEventListener('click', downloadCSV);

// Startup
document.addEventListener('DOMContentLoaded', () => {
    loadDashboardData();
    setInterval(checkHealth, 15000);
});
