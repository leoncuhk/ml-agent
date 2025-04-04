$(document).ready(function() {
    // 导航切换
    $('.nav-link').click(function(e) {
        e.preventDefault();
        $('.nav-link').removeClass('active').addClass('link-dark');
        $(this).addClass('active').removeClass('link-dark');
        
        const pageId = $(this).attr('id').replace('nav-', 'page-');
        $('.page').removeClass('active');
        $('#' + pageId).addClass('active');
    });
    
    // 上传数据文件
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        
        const fileInput = $('#formFile')[0];
        if (fileInput.files.length === 0) {
            alert('请选择文件');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    $('#data-preview-card').show();
                    $('#data-info').html(`<p>行数: ${response.rows}</p>`);
                    $('#data-preview').html(response.preview);
                    
                    // 填充目标变量下拉框
                    const targetSelect = $('#target-var');
                    targetSelect.empty();
                    targetSelect.append('<option selected disabled value="">选择目标变量</option>');
                    
                    response.columns.forEach(function(column) {
                        targetSelect.append(`<option value="${column}">${column}</option>`);
                    });
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('上传失败，请重试');
            }
        });
    });
    
    // 使用示例数据
    $('#use-example').click(function() {
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: new FormData(),
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    $('#data-preview-card').show();
                    $('#data-info').html(`<p>行数: ${response.rows}</p>`);
                    $('#data-preview').html(response.preview);
                    
                    // 填充目标变量下拉框
                    const targetSelect = $('#target-var');
                    targetSelect.empty();
                    targetSelect.append('<option selected disabled value="">选择目标变量</option>');
                    
                    response.columns.forEach(function(column) {
                        targetSelect.append(`<option value="${column}">${column}</option>`);
                    });
                    
                    // 默认选择target
                    targetSelect.val('target');
                    
                    // 填充默认指令
                    $('#instructions').val('请执行分类任务，使用最大运行时间30秒。将target列转换为分类变量。');
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('加载示例数据失败，请重试');
            }
        });
    });
    
    // 训练模型
    $('#train-form').submit(function(e) {
        e.preventDefault();
        
        const targetVar = $('#target-var').val();
        const instructions = $('#instructions').val();
        
        if (!targetVar) {
            alert('请选择目标变量');
            return;
        }
        
        // 显示大模型正在获取建议
        $('#llm-suggestions').html(`
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p>正在获取大模型建议...</p>
        `);
        
        $.ajax({
            url: '/train',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                target: targetVar,
                instructions: instructions
            }),
            success: function(response) {
                if (response.success) {
                    alert(response.message);
                    
                    // 显示训练进度
                    $('#llm-suggestions').html(`
                        <div class="alert alert-info">
                            模型训练已启动，请稍后查看结果...
                        </div>
                        <div id="training-progress">
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" style="width: 100%"></div>
                            </div>
                            <pre id="training-log" class="mt-3"></pre>
                        </div>
                    `);
                    
                    // 轮询训练状态
                    let checkStatusInterval = setInterval(function() {
                        $.ajax({
                            url: '/training-status',
                            type: 'GET',
                            success: function(statusResponse) {
                                // 更新日志信息
                                $('#training-log').text(statusResponse.log_info);
                                
                                // 如果训练完成，停止轮询并检查结果
                                if (statusResponse.status === 'completed') {
                                    clearInterval(checkStatusInterval);
                                    
                                    $('#llm-suggestions').html(`
                                        <div class="alert alert-success">
                                            <strong>训练完成!</strong> 正在加载结果...
                                        </div>
                                    `);
                                    
                                    // 检查训练结果
                                    checkTrainingResults();
                                }
                            },
                            error: function() {
                                $('#training-log').append("\n请求训练状态失败...");
                            }
                        });
                    }, 3000);  // 每3秒检查一次
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('训练请求失败，请重试');
            }
        });
    });
    
    // 检查训练结果
    function checkTrainingResults() {
        $.ajax({
            url: '/results',
            type: 'GET',
            success: function(resultResponse) {
                if (resultResponse.success && resultResponse.best_model_id) {
                    // 显示大模型建议
                    $('#llm-suggestions').html(`
                        <div class="alert alert-success">
                            <strong>训练成功!</strong>
                            <p>最佳模型ID: ${resultResponse.best_model_id}</p>
                        </div>
                        <div class="card mt-3">
                            <div class="card-header">大模型建议</div>
                            <div class="card-body">
                                <pre>${resultResponse.summary}</pre>
                            </div>
                        </div>
                    `);
                    
                    // 自动切换到结果页面
                    $('#nav-results').click();
                } else {
                    // 如果结果还没准备好，5秒后重试
                    setTimeout(checkTrainingResults, 5000);
                }
            },
            error: function() {
                // 如果请求失败，5秒后重试
                setTimeout(checkTrainingResults, 5000);
            }
        });
    }
    
    // 加载结果
    $('#nav-results').click(function() {
        $.ajax({
            url: '/results',
            type: 'GET',
            success: function(response) {
                if (response.success && response.best_model_id) {
                    $('#model-info').html(`
                        <div class="alert alert-success">
                            <p><strong>最佳模型ID:</strong> ${response.best_model_id}</p>
                            <p><strong>模型保存路径:</strong> ${response.model_path}</p>
                            <!-- MLflow Run ID populated here -->
                            <p id="mlflow-run-info"></p> 
                        </div>
                    `);
                    // Populate MLflow info if available
                    if(response.mlflow_run_id) {
                        $('#mlflow-run-info').html(`<strong>MLflow Run ID:</strong> ${response.mlflow_run_id} (<a href="http://localhost:5000/#/experiments/0/runs/${response.mlflow_run_id}" target="_blank">View Run</a>)`);
                    } else {
                        $('#mlflow-run-info').html('MLflow Run ID not available.');
                    }
                    
                    if (response.leaderboard) {
                        $('#leaderboard-container').show();
                        $('#leaderboard').html(response.leaderboard);
                    }
                } else {
                    $('#model-info').html(`
                        <div class="alert alert-warning">
                            <p>${response.error || '还没有训练结果，请先训练模型'}</p>
                        </div>
                    `);
                    $('#leaderboard-container').hide();
                }
            },
            error: function() {
                $('#model-info').html(`
                    <div class="alert alert-danger">
                        <p>加载结果失败，请重试</p>
                    </div>
                `);
                $('#leaderboard-container').hide();
            }
        });
    });
    
    // 模型预测
    $('#predict-btn').click(function() {
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({}),
            success: function(response) {
                if (response.success) {
                    $('#prediction-results').show();
                    $('#predictions').html(response.predictions);
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('预测请求失败，请重试');
            }
        });
    });
    
    // MLflow UI管理
    $('#launch-mlflow-btn').click(function() {
        $.ajax({
            url: '/launch-mlflow',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    alert(response.message);
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('启动MLflow UI失败，请重试');
            }
        });
    });
    
    $('#stop-mlflow-btn').click(function() {
        $.ajax({
            url: '/stop-mlflow',
            type: 'GET',
            success: function(response) {
                if (response.success) {
                    alert(response.message);
                } else {
                    alert('错误: ' + response.error);
                }
            },
            error: function() {
                alert('停止MLflow UI失败，请重试');
            }
        });
    });
    
    // --- Model Interpretation --- 
    // Event listener for the interpret button
    $(document).on('click', '#interpret-btn', function() {
        console.log("Interpret button clicked");
        // Show loading state
        $('#interpretation-plots').html('<div class="spinner-border text-info" role="status"><span class="visually-hidden">Loading...</span></div> <p>Generating interpretations...</p>');
        $('#interpretation-container').show();
        
        $.ajax({
            url: '/interpret',
            type: 'POST',
            contentType: 'application/json', // No data needed for POST, just trigger
            data: JSON.stringify({}), 
            success: function(response) {
                console.log("Interpretation response:", response);
                if (response.success && response.interpretations) {
                    let plotsHtml = '';
                    if (response.interpretations.varimp_plot) {
                        plotsHtml += '<h6>Variable Importance</h6><img src="' + response.interpretations.varimp_plot + '" class="img-fluid mb-3" alt="Variable Importance Plot">';
                    } else if (response.interpretations.varimp_error) {
                         plotsHtml += '<div class="alert alert-warning">Could not generate Variable Importance plot: ' + response.interpretations.varimp_error + '</div>';
                    }
                    
                    // Add placeholders for other plots if needed
                    // if (response.interpretations.shap_plot) { ... }

                    if (plotsHtml === '') {
                         plotsHtml = '<p>No interpretation plots were generated.</p>';
                    }
                    $('#interpretation-plots').html(plotsHtml);
                } else {
                     $('#interpretation-plots').html('<div class="alert alert-danger">Failed to get interpretations: ' + response.error + '</div>');
                }
            },
            error: function(jqXHR, textStatus, errorThrown) {
                 console.error("Interpretation request failed:", textStatus, errorThrown);
                 $('#interpretation-plots').html('<div class="alert alert-danger">Interpretation request failed. Please check server logs.</div>');
            }
        });
    });

    // When results are loaded, show interpret button if successful
    function updateResultsDisplay(response) {
        if (response.success && response.best_model_id) {
            $('#model-info').html(`
                <div class="alert alert-success">
                    <p><strong>最佳模型ID:</strong> ${response.best_model_id}</p>
                    <p><strong>模型保存路径:</strong> ${response.model_path}</p>
                    <!-- MLflow Run ID populated here -->
                    <p id="mlflow-run-info"></p> 
                </div>
            `);
            // Populate MLflow info if available
            if(response.mlflow_run_id) {
                $('#mlflow-run-info').html(`<strong>MLflow Run ID:</strong> ${response.mlflow_run_id} (<a href="http://localhost:5000/#/experiments/0/runs/${response.mlflow_run_id}" target="_blank">View Run</a>)`);
            } else {
                $('#mlflow-run-info').html('MLflow Run ID not available.');
            }
            
            // Show interpret button only if successful and model exists
            $('#interpret-btn').show(); 
            $('#interpretation-container').hide(); // Hide old plots on new result load
            $('#interpretation-plots').html(''); // Clear old plots
            
            if (response.leaderboard) {
                $('#leaderboard-container').show();
                $('#leaderboard').html(response.leaderboard);
            }
        } else {
            $('#model-info').html(`
                <div class="alert alert-warning">
                    <p>${response.error || '还没有训练结果，请先训练模型'}</p>
                </div>
            `);
            $('#leaderboard-container').hide();
            $('#interpret-btn').hide(); // Hide interpret button if no model
            $('#interpretation-container').hide();
            $('#interpretation-plots').html('');
        }
    }
    
    // Modify the existing results loading logic to use the helper function
    $('#nav-results').click(function() {
        $.ajax({
            url: '/results',
            type: 'GET',
            success: function(response) {
                updateResultsDisplay(response); // Use the helper function
            },
            error: function() {
                $('#model-info').html(`
                    <div class="alert alert-danger">
                        <p>加载结果失败，请重试</p>
                    </div>
                `);
                $('#leaderboard-container').hide();
                $('#interpret-btn').hide();
                $('#interpretation-container').hide();
            }
        });
    });

    // Modify checkTrainingResults to potentially call updateResultsDisplay
    function checkTrainingResults() {
        $.ajax({
            url: '/results',
            type: 'GET',
            success: function(resultResponse) {
                if (resultResponse.success && resultResponse.best_model_id) {
                    // ... (existing success message)
                    updateResultsDisplay(resultResponse); // Update display
                    // Switch to results page
                    $('#nav-results').click();
                } else {
                     // Optionally update status message based on error
                    // setTimeout(checkTrainingResults, 5000);
                     console.log("Results not ready yet or training failed:", resultResponse.error);
                     // Stop polling if there was an error reported by /results
                     if (resultResponse.error) {
                          $('#llm-suggestions').html(`<div class="alert alert-danger">Training failed: ${resultResponse.error}</div>`);
                     } else {
                         // Still potentially running, maybe keep polling?
                         setTimeout(checkTrainingResults, 5000);
                     }
                }
            },
            error: function() {
                // setTimeout(checkTrainingResults, 5000);
                 console.log("Failed to fetch results, retrying...");
                 setTimeout(checkTrainingResults, 5000);
            }
        });
    }

    // Initial check for results when page loads (in case state was loaded)
    checkTrainingResults(); // Call once on load

    // --- Prediction --- 
    $('#predict-btn').click(function() {
        // ... (keep existing prediction logic) ...
    });

}); // End document ready