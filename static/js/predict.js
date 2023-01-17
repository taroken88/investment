let ctx = document.getElementById('myChart').getContext('2d');
let title = document.getElementById('chart_title').value;
let target = document.getElementById('chart_target').value;
let dstr = String(document.getElementById('chart_data').value);
let darr = dstr.split(',');
let dlabel = String(document.getElementById('chart_labels').value);
let larr = dlabel.split(',');
let myRadarChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: larr,
        datasets: [{
            data: darr,
            backgroundColor: "rgba(255,0,0,0.2)", // 線の下の塗りつぶしの色
            borderColor: "red",                   // 線の色
            borderWidth: 2,                       // 線の幅
            pointStyle: "circle",                 // 点の形状
            pointRadius: 6,                       // 点形状の半径
            pointBorderColor: "red",              // 点の境界線の色
            pointBorderWidth: 2,                  // 点の境界線の幅
            pointBackgroundColor: "yellow",       // 点の塗りつぶし色
            pointLabelFontSize: 20,
            label: target,
            fill: false
            }
        ]
    },
    // options: {
    //     responsive: true,
    //     title: {                // タイトル
    //         display: true,      // 表示する
    //         fontSize: 20,       // タイトルのフォント
    //         text: title         // 表示するタイトル
    //     },
    //     legend: {
    //         position: 'bottom', // 凡例の表示位置
    //         labels: {
    //             fontSize: 20,   // 判例のフォントサイズ
    //         },
    //     },
    options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: 'Min and Max Settings'
          }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
          }
        }
      },
        // scale: {
            // スケールを隠す。
            // display: true,      // メモリを表示する
            // ticks: {            // 目盛り
            //     min: 0,         // 目盛りの最小値
            //     max: 10        // 目盛りの最大値
                // stepSize: 1,    // 目盛の間隔
                // fontSize: 12,   // 目盛り数字の大きさ
                // fontColor: "purple"  // 目盛り数字の色
            // },
    //         pointLabels: {
    //             fontSize: 20    // チャートラベルのフォントサイズ
    //         },
    //         angleLines: {        // 軸（放射軸）
    //             display: true,
    //             color: "maroon"
    //         },
    //         gridLines: {         // 補助線（目盛の線）
    //             display: true,
    //             color: "lime"
    //         }
        // },
    // }
});