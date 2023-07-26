select
    sites.[objID] as [dc]
    ,year(dateadd(day, 26 - datepart(isoww, fs.[date]), fs.[date])) as [year_iso]
    ,datepart(isowk, fs.[date]) as [week]
    ,datediff(d, '2019-12-30', fs.[date])/7 as [periods]
    ,datediff(d, '2019-12-30', fs.[date])/7%2 as [even_week]
    ,asst.[item_group]
    ,cast(sum(fs.[Количество отгружено]) as float) as pcs_shipped
	,cast(sum(iif(fs.[PROMO] = 1, fs.[Количество отгружено], 0)) as float) as pcs_shipped_PROMO
from [dbo].[shipment] as fs
inner join [dbo].[items] as items on fs.[item_code] = items.[item_code]
inner join [dbo].[assort] as asst on items.[ASSORTCODE] = asst.[LEVELID]
inner join [dbo].[objects] as dcloc on dcloc.[INVENTLOCATIONID]=fs.[РЦ] and dcloc.[FORMAT]='РЦ'
inner join (
	select [SITE], min([objID]) [objID] 
	from [dbo].[objects]
	group by [SITE]) as sites on dcloc.[SITE]=sites.[SITE]
where isnull(fs.[filt1], 0) = 0
    and isnull(fs.[filt2], 0) = 0
    and fs.[date] between '2019-12-30' and dateadd(d, -datepart(dw, dateadd(d, -1, cast(cast(getdate() as date) as date))), cast(getdate() as date))
	and fs.[Количество отгружено] > 0
group by
    year(dateadd(day, 26 - datepart(isoww, fs.[date]), fs.[date]))
    ,datepart(isowk, fs.[date])
    ,datediff(d, '2019-12-30', fs.[date]) / 7
    ,asst.[item_group]
	,sites.[objID]
order by
    year(dateadd(day, 26 - datepart(isoww, fs.[date]), fs.[date]))
    ,datepart(isowk, fs.[date])
    ,datediff(d, '2019-12-30', fs.[date]) / 7
    ,asst.[item_group]
	,sites.[objID]