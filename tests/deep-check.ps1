$apiKey = "sk-32840580d42b43f98dfd4d6a3fe9f9c7"
$url = "https://api.deepseek.com/v1/chat/completions"
$headers = @{
    "Authorization" = "Bearer $apiKey"
    "Content-Type" = "application/json"
}
$body = @{
    "model" = "deepseek-chat"
    "messages" = @(
        @{
            "role" = "user"
            "content" = "Hello!"
        }
    )
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri $url -Method Post -Headers $headers -Body $body
$response.Content